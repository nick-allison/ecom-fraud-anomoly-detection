#!/usr/bin/env python
"""
Train a tabular MLP model on a CSV dataset.

This script demonstrates the end-to-end flow for structured (tabular) data:
- Load configuration (paths, logging, env) via ml_tabular.config
- Load a CSV file into a pandas DataFrame
- Infer / choose problem type (classification vs regression)
- Split into train/validation
- Apply basic numeric scaling
- Wrap data in TabularDataset/DataLoader
- Define a TabularMLP model
- Train with a reusable PyTorch training loop
- Save best model weights and training history

Usage (example):

    python train_tabular_mlp.py \
        --csv-path data/raw/example.csv \
        --target-column target \
        --problem-type auto \
        --epochs 50 \
        --batch-size 256

In a real project you can:
- Move more of these arguments into YAML configs
- Swap the simple scaler for a richer sklearn pipeline
- Add MLflow logging via a ml_tabular.mlops module
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from ml_tabular.config import AppConfig, get_config, get_paths
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger
from ml_tabular.torch_.datasets.tabular import (
    TabularDataset,
    TabularDatasetConfig,
)
from ml_tabular.torch_.models.tabular_mlp import (
    TabularMLP,
    TabularMLPConfig,
)
from ml_tabular.torch_.training.loops import (
    TrainingConfig as TorchTrainingConfig,
    set_seed,
    train_model,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a tabular MLP model on a CSV dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data source
    parser.add_argument(
        "--csv-path",
        type=str,
        required=True,
        help="Path to the input CSV file containing the dataset.",
    )
    parser.add_argument(
        "--target-column",
        type=str,
        required=True,
        help="Name of the target column in the CSV.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default=None,
        help="Optional ID/index column to exclude from features.",
    )

    # Problem semantics
    parser.add_argument(
        "--problem-type",
        type=str,
        choices=["auto", "classification", "regression"],
        default="auto",
        help=(
            "Type of supervised learning problem. "
            "'auto' will infer from the target column type/unique values."
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help=(
            "Number of classes for classification. "
            "If omitted and problem_type='classification', it will be inferred "
            "from the unique values of the target column."
        ),
    )

    # Model hyperparameters
    parser.add_argument(
        "--hidden-dims",
        type=str,
        default="128,64",
        help="Comma-separated hidden layer sizes for the MLP, e.g. '256,128,64'.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability applied after hidden layers.",
    )
    parser.add_argument(
        "--activation",
        type=str,
        choices=["relu", "gelu"],
        default="relu",
        help="Activation function used in the hidden layers.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training and validation.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate for the optimizer.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (L2 regularization) for the optimizer.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to train on: 'cpu', 'cuda', or 'auto' to pick automatically.",
    )

    # Config integration
    parser.add_argument(
        "--config-path",
        type=str,
        default=None,
        help=(
            "Optional path to a YAML config file. "
            "If provided, it will be loaded by ml_tabular.config.get_config "
            "to set paths, env, log_level, etc."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default=None,
        help=(
            "Optional environment name (e.g. 'dev', 'prod') used by the config "
            "system to select a profile in the YAML config."
        ),
    )

    # Experiment naming / output control
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help=(
            "Optional experiment name. If omitted, AppConfig.experiment_name is used. "
            "Used to name the output directory under the models path."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_device(device_arg: str) -> str:
    """Resolve 'auto' device choice."""
    import torch

    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg in {"cpu", "cuda"}:
        return device_arg
    raise ValueError(f"Unsupported device argument: {device_arg!r}")


def _infer_problem_type_and_classes(
    df: pd.DataFrame,
    target_column: str,
    requested: Literal["auto", "classification", "regression"],
    num_classes: Optional[int],
) -> tuple[str, Optional[int]]:
    """
    Decide problem type and number of classes (if classification).

    Heuristics when requested='auto':
      - If target dtype is object/category/bool, treat as classification.
      - If numeric with few unique values (<= 20), treat as classification.
      - Otherwise treat as regression.
    """
    series = df[target_column]

    if requested in {"classification", "regression"}:
        problem_type = requested
    else:
        # auto mode
        if series.dtype == "bool" or str(series.dtype) in ("category", "object"):
            problem_type = "classification"
        else:
            # numeric-like
            unique_values = series.dropna().unique()
            if series.dtype.kind in "iu" and len(unique_values) <= 20:
                problem_type = "classification"
            else:
                problem_type = "regression"

    inferred_num_classes: Optional[int] = None
    if problem_type == "classification":
        inferred_num_classes = int(series.nunique()) if num_classes is None else num_classes

    return problem_type, inferred_num_classes


def _prepare_features_and_target(
    df: pd.DataFrame,
    target_column: str,
    id_column: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target, dropping optional ID column."""
    if target_column not in df.columns:
        raise DataError(
            f"Target column '{target_column}' not found in CSV.",
            code="missing_target_column",
            context={"available_columns": list(df.columns), "target_column": target_column},
            location="train_tabular_mlp._prepare_features_and_target",
        )

    if id_column is not None and id_column not in df.columns:
        raise DataError(
            f"ID column '{id_column}' not found in CSV.",
            code="missing_id_column",
            context={"available_columns": list(df.columns), "id_column": id_column},
            location="train_tabular_mlp._prepare_features_and_target",
        )

    y = df[target_column]
    feature_cols = [c for c in df.columns if c != target_column and c != id_column]
    X = df[feature_cols]

    return X, y


def _select_numeric_features(X: pd.DataFrame) -> pd.DataFrame:
    """Return only numeric feature columns, dropping non-numeric with a warning."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        raise DataError(
            "No numeric feature columns found after selection.",
            code="no_numeric_features",
            context={"columns": list(X.columns)},
            location="train_tabular_mlp._select_numeric_features",
        )
    non_numeric = sorted(set(X.columns) - set(numeric_cols))
    if non_numeric:
        logger = get_logger(__name__)
        logger.warning(
            "Dropping non-numeric feature columns for MLP: %s", ", ".join(non_numeric)
        )
    return X[numeric_cols]


def _build_scaler_and_transform(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit a StandardScaler on training features and transform both train/val.

    In a more advanced pipeline you could:
    - Handle categorical features via OneHotEncoder,
    - Chain imputation + scaling via ColumnTransformer,
    - Persist the fitted preprocessor for inference.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.to_numpy())
    X_val_scaled = scaler.transform(X_val.to_numpy())
    return X_train_scaled, X_val_scaled, scaler


# ---------------------------------------------------------------------------
# Main training flow
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Load application config (paths, env, logging level, etc.)
    cfg: AppConfig = get_config(
        config_path=args.config_path,
        env=args.env,
    )
    paths = get_paths(config_path=args.config_path, env=args.env)

    # Initialize logging
    logger = get_logger(__name__)
    logger.info("Starting tabular MLP training script.")
    logger.info("Environment: %s", cfg.env)
    logger.info("CSV path: %s", args.csv_path)

    # Resolve CSV path: relative paths interpreted relative to data_dir
    csv_path = Path(args.csv_path)
    if not csv_path.is_absolute():
        csv_path = (paths.data_dir / csv_path).resolve()
    if not csv_path.exists():
        raise DataError(
            f"CSV file not found: {csv_path}",
            code="csv_not_found",
            context={"csv_path": str(csv_path)},
            location="train_tabular_mlp.main",
        )

    # Load data
    logger.info("Loading data from %s", csv_path)
    df = pd.read_csv(csv_path)
    logger.info("Loaded DataFrame with shape: %s", df.shape)

    # Split into features + target
    X, y = _prepare_features_and_target(
        df=df,
        target_column=args.target_column,
        id_column=args.id_column,
    )
    logger.info("Feature matrix shape before selection: %s", X.shape)

    # Keep only numeric features for this baseline MLP
    X = _select_numeric_features(X)
    logger.info("Feature matrix shape after numeric selection: %s", X.shape)

    # Infer or enforce problem type and num_classes
    problem_type, num_classes = _infer_problem_type_and_classes(
        df=df,
        target_column=args.target_column,
        requested=args.problem_type,  # 'auto', 'classification', or 'regression'
        num_classes=args.num_classes,
    )

    if problem_type == "classification":
        if num_classes is None:
            raise DataError(
                "Classification problem but num_classes could not be inferred.",
                code="num_classes_missing",
                context={"target_column": args.target_column},
                location="train_tabular_mlp.main",
            )
        logger.info(
            "Problem type: classification (num_classes=%d)", num_classes
        )
    else:
        logger.info("Problem type: regression")

    # Train/validation split (uses config.training.test_size and random_seed)
    test_size = cfg.training.test_size
    random_seed = cfg.training.random_seed
    logger.info("Splitting data: test_size=%.3f, random_seed=%d", test_size, random_seed)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y if problem_type == "classification" else None,
    )

    logger.info("Train shapes: X=%s, y=%s", X_train.shape, y_train.shape)
    logger.info("Val   shapes: X=%s, y=%s", X_val.shape, y_val.shape)

    # Scale numeric features
    X_train_scaled, X_val_scaled, scaler = _build_scaler_and_transform(X_train, X_val)
    logger.info("Applied StandardScaler to numeric features.")

    # Convert targets to numpy arrays
    y_train_np = y_train.to_numpy()
    y_val_np = y_val.to_numpy()

    # Build PyTorch datasets
    ds_config = TabularDatasetConfig(
        problem_type=problem_type,
        num_classes=num_classes,
    )

    train_dataset = TabularDataset(
        X=X_train_scaled,
        y=y_train_np,
        config=ds_config,
    )
    val_dataset = TabularDataset(
        X=X_val_scaled,
        y=y_val_np,
        config=ds_config,
    )

    # Build DataLoaders
    import torch
    from torch.utils.data import DataLoader

    batch_size = args.batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    logger.info("Created DataLoaders with batch_size=%d", batch_size)

    # Model configuration
    hidden_dims = [int(h.strip()) for h in args.hidden_dims.split(",") if h.strip()]
    if not hidden_dims:
        raise ValueError("At least one hidden dimension must be provided in --hidden-dims.")

    input_dim = X_train_scaled.shape[1]
    output_dim = num_classes if problem_type == "classification" else 1

    model_config = TabularMLPConfig(
        input_dim=input_dim,
        output_dim=output_dim,
        problem_type=problem_type,
        hidden_dims=hidden_dims,
        dropout=args.dropout,
        activation=args.activation,
    )

    model = TabularMLP(config=model_config)
    logger.info(
        "Initialized TabularMLP: input_dim=%d, output_dim=%d, hidden_dims=%s, "
        "dropout=%.3f, activation=%s",
        input_dim,
        output_dim,
        hidden_dims,
        args.dropout,
        args.activation,
    )

    # Training configuration for the PyTorch loop
    device = _resolve_device(args.device)
    torch_cfg = TorchTrainingConfig(
        problem_type=problem_type,
        num_classes=num_classes,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=batch_size,
        seed=random_seed,
        device=device,
        log_every_n_batches=10,
    )

    logger.info(
        "Training config: epochs=%d, lr=%s, weight_decay=%s, device=%s",
        torch_cfg.epochs,
        torch_cfg.learning_rate,
        torch_cfg.weight_decay,
        torch_cfg.device,
    )

    # Seed everything for reproducibility
    set_seed(torch_cfg.seed)

    # Move model to device
    model.to(device)

    # Train
    logger.info("Starting training...")
    history, best_state_dict = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=torch_cfg,
    )
    logger.info("Training finished.")

    # Save artifacts
    experiment_name = args.experiment_name or cfg.experiment_name
    output_dir = (paths.models_dir / experiment_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model_best.pt"
    torch.save(best_state_dict, model_path)

    history_path = output_dir / "training_history.csv"
    history.to_csv(history_path, index=False)

    logger.info("Saved best model weights to: %s", model_path)
    logger.info("Saved training history to: %s", history_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
