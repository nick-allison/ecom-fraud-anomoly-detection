#!/usr/bin/env python
"""
Train a time-series MLP using the ml_tabular template.

This script expects a YAML config (e.g. configs/time_series/train_ts_baseline.yaml)
and uses:

- ml_tabular.config.get_config / get_paths
- ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset
- ml_tabular.torch.models.tabular_mlp.TabularMLP / TabularMLPConfig
- ml_tabular.torch.training.loops.fit / EarlyStopping
- ml_tabular.mlops.mlflow_utils for optional MLflow logging

It is written for *time-series regression* (forecasting). Extension points
for classification are noted in comments where relevant.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader

from ml_tabular import (
    EarlyStopping,
    TabularMLP,
    TabularMLPConfig,
    TimeSeriesSequenceDataset,
    fit,
    get_config,
    get_logger,
    get_paths,
)
from ml_tabular.mlops.mlflow_utils import (
    log_params as mlflow_log_params,
    mlflow_is_enabled,
    mlflow_run,
)

# ---------------------------------------------------------------------------
# CLI + utilities
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a time-series MLP using the ml_tabular template.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="configs/time_series/train_ts_baseline.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help="Config environment/profile name (e.g. 'dev', 'prod'). "
        "If omitted, the default from ml_tabular.config is used.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to use: 'auto' (CUDA if available, else CPU), 'cpu', or 'cuda'.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Disable MLflow logging even if configured.",
    )
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: --device cuda requested but no CUDA available; falling back to CPU.")
        return torch.device("cpu")

    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # You *may* want full determinism in some cases:
    # torch.use_deterministic_algorithms(True, warn_only=True)


# ---------------------------------------------------------------------------
# Loss + metrics for time-series regression
# ---------------------------------------------------------------------------


class TimeSeriesMSELoss(nn.Module):
    """MSE loss for time-series windows.

    Flattens (B, H, D_out) and (B, output_dim) into (B, -1) and computes MSE.
    This lets us use TabularMLP (which outputs 2D) with sequence targets.
    """

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        # preds: (B, output_dim)
        # targets: (B, H, D_out)
        preds_flat = preds.view(preds.shape[0], -1)
        targets_flat = targets.view(targets.shape[0], -1)
        return F.mse_loss(preds_flat, targets_flat)


def mae_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Mean Absolute Error (MAE) for time-series forecasts."""
    y_true_flat = y_true.view(y_true.shape[0], -1)
    y_pred_flat = y_pred.view(y_pred.shape[0], -1)
    return torch.mean(torch.abs(y_true_flat - y_pred_flat)).item()


def rmse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Root Mean Squared Error (RMSE) for time-series forecasts."""
    y_true_flat = y_true.view(y_true.shape[0], -1)
    y_pred_flat = y_pred.view(y_pred.shape[0], -1)
    mse = torch.mean((y_true_flat - y_pred_flat) ** 2)
    return torch.sqrt(mse).item()


# ---------------------------------------------------------------------------
# Data loading / splitting
# ---------------------------------------------------------------------------


def build_datasets_from_config(
    cfg: Any,
    paths: Any,
    logger: Any,
) -> tuple[TimeSeriesSequenceDataset, TimeSeriesSequenceDataset]:
    """Build train/val TimeSeriesSequenceDataset from config + paths.

    Expected config structure (you can adjust names to match your actual Pydantic models):

      cfg.time_series:
        dataset_csv: "time_series.csv"     # relative to paths.data_dir
        datetime_column: "timestamp"       # column to sort by
        target_column: "target"            # name of target column
        value_columns: ["feat1", "feat2"]  # optional; if omitted, infer
        val_fraction: 0.2                  # fraction of data reserved for validation
        input_window: 48
        prediction_horizon: 12
        step_size: 1
        task_type: "regression"            # currently used for sanity checks

    This function:

      - Reads the CSV from data_dir / dataset_csv.
      - Sorts by datetime_column (if provided).
      - Splits by time into train/val according to val_fraction.
      - Builds TimeSeriesSequenceDataset for each split.
    """
    ts_cfg = cfg.time_series
    data_dir = Path(paths.data_dir)
    csv_path = data_dir / ts_cfg.dataset_csv

    if not csv_path.exists():
        raise FileNotFoundError(f"Time-series dataset not found at: {csv_path}")

    logger.info("Loading time-series data from %s", csv_path)
    df = pd.read_csv(csv_path)

    if getattr(ts_cfg, "datetime_column", None):
        dt_col = ts_cfg.datetime_column
        if dt_col not in df.columns:
            raise ValueError(
                f"Configured datetime_column '{dt_col}' not found in data columns: "
                f"{list(df.columns)}"
            )
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.sort_values(by=dt_col)

    target_col = ts_cfg.target_column
    if target_col not in df.columns:
        raise ValueError(
            f"Configured target_column '{target_col}' not found in data columns: "
            f"{list(df.columns)}"
        )

    if getattr(ts_cfg, "value_columns", None):
        value_columns = list(ts_cfg.value_columns)
    else:
        # Use all columns except datetime + target as features
        exclude = {target_col}
        if getattr(ts_cfg, "datetime_column", None):
            exclude.add(ts_cfg.datetime_column)
        value_columns = [c for c in df.columns if c not in exclude]

    if not value_columns:
        raise ValueError("No value_columns could be inferred for time-series features.")

    val_fraction = float(getattr(ts_cfg, "val_fraction", 0.2))
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("time_series.val_fraction must be in (0, 1).")

    n_total = len(df)
    n_val = max(1, int(round(n_total * val_fraction)))
    if n_val >= n_total:
        raise ValueError("Validation set size is >= total dataset size.")

    # Time-based split: earliest data for training, most recent for validation
    train_df = df.iloc[: n_total - n_val]
    val_df = df.iloc[n_total - n_val :]

    logger.info(
        "Time-series split: total=%d, train=%d, val=%d (val_fraction=%.3f)",
        n_total,
        len(train_df),
        len(val_df),
        val_fraction,
    )

    train_ds = TimeSeriesSequenceDataset.from_dataframe(
        train_df,
        value_columns=value_columns,
        target_column=target_col,
        sort_by=ts_cfg.datetime_column if getattr(ts_cfg, "datetime_column", None) else None,
        dropna=True,
        input_window=ts_cfg.input_window,
        prediction_horizon=ts_cfg.prediction_horizon,
        step_size=ts_cfg.step_size,
    )

    val_ds = TimeSeriesSequenceDataset.from_dataframe(
        val_df,
        value_columns=value_columns,
        target_column=target_col,
        sort_by=ts_cfg.datetime_column if getattr(ts_cfg, "datetime_column", None) else None,
        dropna=True,
        input_window=ts_cfg.input_window,
        prediction_horizon=ts_cfg.prediction_horizon,
        step_size=ts_cfg.step_size,
    )

    logger.info(
        "Train dataset: n_samples=%d, input_window=%d, horizon=%d, features=%d, targets=%d",
        len(train_ds),
        train_ds.metadata.input_window,
        train_ds.metadata.prediction_horizon,
        train_ds.metadata.num_features,
        train_ds.metadata.num_targets,
    )
    logger.info(
        "Val dataset:   n_samples=%d, input_window=%d, horizon=%d, features=%d, targets=%d",
        len(val_ds),
        val_ds.metadata.input_window,
        val_ds.metadata.prediction_horizon,
        val_ds.metadata.num_features,
        val_ds.metadata.num_targets,
    )

    return train_ds, val_ds


# ---------------------------------------------------------------------------
# Model + optimizer + scheduler from config
# ---------------------------------------------------------------------------


def build_model_from_config(
    cfg: Any,
    train_ds: TimeSeriesSequenceDataset,
    logger: Any,
) -> TabularMLP:
    """Build a TabularMLP for time-series forecasting.

    We treat each input window (W, D_in) as a flattened vector of size
    W * D_in, and each target window (H, D_out) as a flattened vector of
    size H * D_out. The MLP is a pure feed-forward network:

        x: (B, W, D_in) -> flatten -> (B, W * D_in) -> MLP -> (B, H * D_out)

    Expected config fields (you can adapt names as needed):

      cfg.time_series.task_type: "regression" (currently assumed)
      cfg.training.hidden_dims: [256, 128]      (optional)
      cfg.training.activation: "relu"           (optional)
      cfg.training.dropout: 0.1                 (optional)
      cfg.training.batch_norm: true             (optional)
      cfg.training.layer_norm: false            (optional)
    """
    ts_cfg = cfg.time_series
    train_cfg = cfg.training

    # For now we treat time series as regression
    task_type = getattr(ts_cfg, "task_type", "regression")
    if task_type != "regression":
        logger.warning(
            "time_series.task_type=%r; current script assumes regression. "
            "You may need to customize the model/loss for classification tasks.",
            task_type,
        )

    meta = train_ds.metadata
    input_dim = meta.input_window * meta.num_features
    if meta.num_targets <= 0:
        raise ValueError("TimeSeriesSequenceDataset has no targets; cannot train supervised model.")
    output_dim = meta.prediction_horizon * meta.num_targets

    hidden_dims = getattr(train_cfg, "hidden_dims", (256, 128))
    activation = getattr(train_cfg, "activation", "relu")
    dropout = float(getattr(train_cfg, "dropout", 0.1))
    batch_norm = bool(getattr(train_cfg, "batch_norm", True))
    layer_norm = bool(getattr(train_cfg, "layer_norm", False))

    mlp_config = TabularMLPConfig(
        input_dim=input_dim,
        task_type="regression",  # force regression semantics here
        hidden_dims=tuple(hidden_dims),
        activation=activation,
        dropout=dropout,
        batch_norm=batch_norm,
        layer_norm=layer_norm,
        output_dim=output_dim,
    )

    model = TabularMLP(mlp_config)
    logger.info(
        "Built TabularMLP for time-series: input_dim=%d, output_dim=%d, hidden_dims=%s",
        input_dim,
        output_dim,
        list(hidden_dims),
    )
    return model


def build_optimizer_and_scheduler_from_config(
    cfg: Any,
    model: nn.Module,
) -> tuple[torch.optim.Optimizer, Any]:
    """Build optimizer and optional scheduler from cfg.training.

    Expected fields (adapt as needed):

      cfg.training.learning_rate: float
      cfg.training.weight_decay: float
      cfg.training.optimizer: "adamw" | "adam" | "sgd"
      cfg.training.scheduler_name: "none" | "reduce_on_plateau" | "cosine"
      cfg.training.scheduler_factor: float
      cfg.training.scheduler_patience: int
      cfg.training.scheduler_min_lr: float
      cfg.training.scheduler_T_max: int
    """
    train_cfg = cfg.training

    lr = float(getattr(train_cfg, "learning_rate", 1e-3))
    weight_decay = float(getattr(train_cfg, "weight_decay", 1e-4))
    optimizer_name = str(getattr(train_cfg, "optimizer", "adamw")).lower()

    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(getattr(train_cfg, "momentum", 0.9)),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name!r}")

    scheduler_name = str(getattr(train_cfg, "scheduler_name", "none")).lower()
    scheduler = None

    if scheduler_name == "none":
        scheduler = None
    elif scheduler_name == "reduce_on_plateau":
        factor = float(getattr(train_cfg, "scheduler_factor", 0.1))
        patience = int(getattr(train_cfg, "scheduler_patience", 5))
        min_lr = float(getattr(train_cfg, "scheduler_min_lr", 1e-6))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )
    elif scheduler_name == "cosine":
        # Cosine decay over a fixed horizon (T_max epochs)
        T_max = int(getattr(train_cfg, "scheduler_T_max", 10))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unsupported scheduler_name: {scheduler_name!r}")

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Main training entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Config + logging
    cfg = get_config(config_path=args.config, env=args.env)
    paths = get_paths(config_path=args.config, env=args.env)
    logger = get_logger("train_ts_mlp")

    logger.info("Loaded config from %s (env=%r)", args.config, args.env)

    # Seeding
    seed = int(getattr(cfg.training, "random_seed", 42))
    set_global_seed(seed)
    logger.info("Random seed set to %d", seed)

    # Device
    device = select_device(args.device)
    logger.info("Using device: %s", device)

    # Datasets + loaders
    train_ds, val_ds = build_datasets_from_config(cfg, paths, logger)

    batch_size = int(getattr(cfg.training, "batch_size", 64))
    num_workers = int(getattr(cfg.training, "num_workers", 0))
    pin_memory = bool(device.type == "cuda")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # Model, optimizer, scheduler, loss, metrics
    model = build_model_from_config(cfg, train_ds, logger)
    optimizer, scheduler = build_optimizer_and_scheduler_from_config(cfg, model)
    loss_fn = TimeSeriesMSELoss()

    metrics: Dict[str, Any] = {
        "mae": mae_metric,
        "rmse": rmse_metric,
    }

    num_epochs = int(getattr(cfg.training, "num_epochs", 50))
    use_amp = bool(getattr(cfg.training, "use_amp", True))
    max_grad_norm = float(getattr(cfg.training, "max_grad_norm", 1.0))  # or None
    if max_grad_norm <= 0.0:
        max_grad_norm = None
    grad_accum = int(getattr(cfg.training, "gradient_accumulation_steps", 1))
    log_interval = getattr(cfg.training, "log_interval", None)

    early_stopping_enabled = bool(getattr(cfg.training, "early_stopping_enabled", True))
    early_stopping = None
    if early_stopping_enabled:
        early_stopping = EarlyStopping(
            monitor=str(getattr(cfg.training, "early_stopping_monitor", "val_loss")),
            mode=str(getattr(cfg.training, "early_stopping_mode", "min")),
            patience=int(getattr(cfg.training, "early_stopping_patience", 10)),
            min_delta=float(getattr(cfg.training, "early_stopping_min_delta", 0.0)),
        )

    scaler = GradScaler() if (use_amp and device.type == "cuda") else None

    # MLflow
    mlflow_enabled = (not args.no_mlflow) and mlflow_is_enabled()
    logger.info("MLflow logging enabled: %s", mlflow_enabled)

    # Prepare model directory
    models_dir = Path(paths.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    experiment_name = getattr(cfg, "experiment_name", "ts_mlp_experiment")
    model_path = models_dir / f"{experiment_name}_ts_mlp.pt"

    # Flatten some key config fields for MLflow params
    params: Dict[str, Any] = {
        "seed": seed,
        "device": str(device),
        "batch_size": batch_size,
        "num_epochs": num_epochs,
        "learning_rate": float(getattr(cfg.training, "learning_rate", 1e-3)),
        "weight_decay": float(getattr(cfg.training, "weight_decay", 1e-4)),
        "optimizer": str(getattr(cfg.training, "optimizer", "adamw")),
        "scheduler_name": str(getattr(cfg.training, "scheduler_name", "none")),
        "ts_input_window": int(train_ds.metadata.input_window),
        "ts_prediction_horizon": int(train_ds.metadata.prediction_horizon),
        "ts_num_features": int(train_ds.metadata.num_features),
        "ts_num_targets": int(train_ds.metadata.num_targets),
    }

    with mlflow_run(enabled=mlflow_enabled) as _run:
        # Log params once per run
        if mlflow_enabled:
            mlflow_log_params(params)

        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            num_epochs=num_epochs,
            metrics=metrics,
            scheduler=scheduler,
            scheduler_on="val_metric" if scheduler is not None else "epoch",
            scheduler_metric="val_loss",
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=grad_accum,
            log_interval=log_interval,
            early_stopping=early_stopping,
            use_mlflow=mlflow_enabled,
        )

        logger.info("Training finished. Last-epoch metrics: %s", history["last_epoch"])

        # Save model weights locally
        torch.save(model.state_dict(), model_path)
        logger.info("Saved model state_dict to %s", model_path)

        # If you later add a log_artifact helper, you can log the model file here.

    logger.info("train_ts_mlp.py completed successfully.")


if __name__ == "__main__":
    main()
