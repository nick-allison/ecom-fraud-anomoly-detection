from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ml_tabular.exceptions import DataError, TrainingError
from ml_tabular.logging_config import get_logger
from ml_tabular.mlops.mlflow_utils import (
    log_metrics as mlflow_log_metrics,
    mlflow_is_enabled,
)

logger = get_logger(__name__)

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------


@dataclass
class EarlyStopping:
    """Simple early-stopping utility for validation metrics.

    Parameters
    ----------
    monitor:
        Name of the metric being monitored (e.g., "val_loss" or "val_accuracy").
    mode:
        "min" -> lower is better (e.g., loss).
        "max" -> higher is better (e.g., accuracy).
    patience:
        Number of epochs with no improvement after which training is stopped.
    min_delta:
        Minimum absolute change to qualify as an improvement.
    """

    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    min_delta: float = 0.0

    best_score: Optional[float] = None
    num_bad_epochs: int = 0
    should_stop: bool = False

    def __post_init__(self) -> None:
        mode_lower = self.mode.lower()
        if mode_lower not in {"min", "max"}:
            raise TrainingError(
                f"Invalid mode for EarlyStopping: {self.mode!r}. Must be 'min' or 'max'.",
                code="early_stopping_bad_mode",
                context={"mode": self.mode},
                location="ml_tabular.torch.training.loops.EarlyStopping.__post_init__",
            )
        self.mode = mode_lower

    def _is_improvement(self, current: float, best: float) -> bool:
        if self.mode == "min":
            return (best - current) > self.min_delta
        return (current - best) > self.min_delta

    def step(self, metrics: Mapping[str, float]) -> None:
        """Update the early-stopping state with metrics from the latest epoch."""
        if self.monitor not in metrics:
            raise TrainingError(
                f"Metric '{self.monitor}' not found in metrics dict.",
                code="early_stopping_missing_metric",
                context={"available_metrics": list(metrics.keys())},
                location="ml_tabular.torch.training.loops.EarlyStopping.step",
            )

        current = float(metrics[self.monitor])

        if self.best_score is None:
            self.best_score = current
            self.num_bad_epochs = 0
            logger.info(
                "EarlyStopping init: monitor=%s, best_score=%.6f",
                self.monitor,
                self.best_score,
            )
            return

        if self._is_improvement(current, self.best_score):
            self.best_score = current
            self.num_bad_epochs = 0
            logger.info(
                "EarlyStopping: improvement detected on %s, new best_score=%.6f",
                self.monitor,
                self.best_score,
            )
        else:
            self.num_bad_epochs += 1
            logger.info(
                "EarlyStopping: no improvement on %s (current=%.6f, best=%.6f). "
                "num_bad_epochs=%d/%d",
                self.monitor,
                current,
                self.best_score,
                self.num_bad_epochs,
                self.patience,
            )

            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
                logger.info(
                    "EarlyStopping: stopping criterion met. "
                    "No improvement in %d epochs on '%s'.",
                    self.patience,
                    self.monitor,
                )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Move a batch from a DataLoader to the specified device.

    Expects:
      - batch to be (inputs, targets), or
      - batch to be inputs only (targets=None).

    Returns
    -------
    (inputs, targets)
        Both tensors on the specified device.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) == 2:
            x, y = batch
        else:
            raise TrainingError(
                "Expected batch to be (inputs, targets) or inputs only.",
                code="batch_bad_structure",
                context={"len_batch": len(batch)},
                location="ml_tabular.torch.training.loops._move_batch_to_device",
            )
    else:
        x, y = batch, None

    x = x.to(device)
    if y is not None:
        y = y.to(device)

    return x, y


def _update_metric_accumulators(
    metric_fns: Mapping[str, MetricFn],
    accumulators: MutableMapping[str, float],
    counts: MutableMapping[str, int],
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
) -> None:
    """Update running sums for metric values.

    Each metric is assumed to produce a scalar float for the provided batch.
    We accumulate as a *weighted average* by number of samples in the batch.
    """
    batch_size = int(y_true.shape[0])
    y_true_cpu = y_true.detach().cpu()
    y_pred_cpu = y_pred.detach().cpu()

    for name, fn in metric_fns.items():
        try:
            value = float(fn(y_true_cpu, y_pred_cpu))
        except Exception as exc:
            raise TrainingError(
                f"Error while computing metric '{name}'.",
                code="metric_computation_error",
                cause=exc,
                context={"metric_name": name},
                location="ml_tabular.torch.training.loops._update_metric_accumulators",
            ) from exc

        accumulators[name] = accumulators.get(name, 0.0) + value * batch_size
        counts[name] = counts.get(name, 0) + batch_size


def _finalize_metrics(
    loss_sum: float,
    n_samples: int,
    metric_accumulators: Mapping[str, float],
    metric_counts: Mapping[str, int],
) -> Dict[str, float]:
    """Compute average loss and metrics over an epoch."""
    if n_samples <= 0:
        raise TrainingError(
            "No samples were seen during the loop. "
            "Check that your DataLoader is not empty.",
            code="no_samples_seen",
            context={},
            location="ml_tabular.torch.training.loops._finalize_metrics",
        )

    results: Dict[str, float] = {}
    results["loss"] = loss_sum / n_samples

    for name, total in metric_accumulators.items():
        count = metric_counts.get(name, 0)
        if count == 0:
            continue
        results[name] = total / count

    return results


# ---------------------------------------------------------------------------
# Public training/evaluation loops
# ---------------------------------------------------------------------------


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    metrics: Optional[Mapping[str, MetricFn]] = None,
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    log_interval: Optional[int] = None,
) -> Dict[str, float]:
    """Run a single training epoch.

    Parameters
    ----------
    model:
        The model to train (must be on the given device).
    dataloader:
        DataLoader yielding (inputs, targets) batches.
    optimizer:
        Optimizer instance.
    loss_fn:
        Loss function: takes (predictions, targets) and returns a scalar tensor.
    device:
        torch.device to run on.
    metrics:
        Optional mapping of metric name -> callable(y_true, y_pred) returning a float.
        Metrics are computed on CPU with detached tensors.
    scaler:
        Optional GradScaler for mixed-precision training (AMP). If provided and
        device.type == "cuda", autocast() + scaler is used.
    max_grad_norm:
        If provided, gradients are clipped to this norm (before optimizer.step()).
    gradient_accumulation_steps:
        Accumulate gradients over this many batches before optimizer.step().
    log_interval:
        If provided, log progress every `log_interval` batches.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "loss" and any metric names, averaged over the epoch.
    """
    model.train()

    metric_fns = metrics or {}
    loss_sum = 0.0
    n_samples = 0

    metric_accumulators: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    use_amp = scaler is not None and device.type == "cuda"
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader, start=1):
        x, y = _move_batch_to_device(batch, device)

        if y is None:
            raise TrainingError(
                "train_one_epoch expected (inputs, targets) but got targets=None.",
                code="train_missing_targets",
                context={},
                location="ml_tabular.torch.training.loops.train_one_epoch",
            )

        batch_size = int(x.shape[0])

        if use_amp:
            with autocast():
                preds = model(x)
                loss = loss_fn(preds, y) / gradient_accumulation_steps
        else:
            preds = model(x)
            loss = loss_fn(preds, y) / gradient_accumulation_steps

        if not torch.isfinite(loss):
            raise TrainingError(
                "Non-finite loss encountered during training.",
                code="non_finite_loss",
                context={"loss": float(loss.detach().cpu().item())},
                location="ml_tabular.torch.training.loops.train_one_epoch",
            )

        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update accumulators
        loss_sum += float(loss.detach().cpu().item()) * batch_size * gradient_accumulation_steps
        n_samples += batch_size

        if metric_fns:
            _update_metric_accumulators(metric_fns, metric_accumulators, metric_counts, y, preds)

        # Gradient step (with optional accumulation)
        if batch_idx % gradient_accumulation_steps == 0:
            if use_amp:
                if max_grad_norm is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        if log_interval is not None and batch_idx % log_interval == 0:
            logger.info(
                "Train batch %d: loss=%.6f",
                batch_idx,
                float(loss.detach().cpu().item() * gradient_accumulation_steps),
            )

    return _finalize_metrics(loss_sum, n_samples, metric_accumulators, metric_counts)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    metrics: Optional[Mapping[str, MetricFn]] = None,
) -> Dict[str, float]:
    """Evaluate a model on a validation/test DataLoader.

    Parameters
    ----------
    model:
        The model to evaluate (must be on the given device).
    dataloader:
        DataLoader yielding (inputs, targets) batches.
    loss_fn:
        Loss function: takes (predictions, targets) and returns a scalar tensor.
    device:
        torch.device to run on.
    metrics:
        Optional mapping of metric name -> callable(y_true, y_pred) returning a float.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys "loss" and any metric names, averaged over all batches.
    """
    model.eval()

    metric_fns = metrics or {}
    loss_sum = 0.0
    n_samples = 0
    metric_accumulators: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    with torch.no_grad():
        for batch in dataloader:
            x, y = _move_batch_to_device(batch, device)

            if y is None:
                raise TrainingError(
                    "evaluate expected (inputs, targets) but got targets=None.",
                    code="eval_missing_targets",
                    context={},
                    location="ml_tabular.torch.training.loops.evaluate",
                )

            preds = model(x)
            loss = loss_fn(preds, y)

            batch_size = int(x.shape[0])
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            n_samples += batch_size

            if metric_fns:
                _update_metric_accumulators(metric_fns, metric_accumulators, metric_counts, y, preds)

    return _finalize_metrics(loss_sum, n_samples, metric_accumulators, metric_counts)


# ---------------------------------------------------------------------------
# High-level fit loop
# ---------------------------------------------------------------------------


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    num_epochs: int,
    metrics: Optional[Mapping[str, MetricFn]] = None,
    scheduler: Optional[Any] = None,
    scheduler_on: str = "epoch",  # "epoch" or "val_metric"
    scheduler_metric: str = "val_loss",
    scaler: Optional[GradScaler] = None,
    max_grad_norm: Optional[float] = None,
    gradient_accumulation_steps: int = 1,
    log_interval: Optional[int] = None,
    early_stopping: Optional[EarlyStopping] = None,
    use_mlflow: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Train a model for multiple epochs with optional validation, scheduler, and MLflow.

    Parameters
    ----------
    model, train_loader, optimizer, loss_fn, device:
        Standard training components.
    num_epochs:
        Number of epochs to train for.
    metrics:
        Optional mapping of metric name -> callable(y_true, y_pred) for both
        training and validation.
    scheduler:
        Optional LR scheduler. If provided:
          - If scheduler_on == "epoch", scheduler.step() is called after each epoch.
          - If scheduler_on == "val_metric" and val_loader is not None, we call
            scheduler.step(val_metrics[scheduler_metric]).
    scheduler_on:
        "epoch" or "val_metric". Controls how scheduler.step() is called.
    scheduler_metric:
        Metric name used when scheduler_on == "val_metric".
    scaler:
        Optional GradScaler for AMP.
    max_grad_norm:
        Optional gradient clipping norm.
    gradient_accumulation_steps:
        Accumulate gradients over this many batches in train_one_epoch.
    log_interval:
        Passed through to train_one_epoch for batch-level logging.
    early_stopping:
        Optional EarlyStopping instance. If provided and should_stop becomes True,
        training stops early.
    use_mlflow:
        If True and MLflow is enabled in config, per-epoch metrics are logged
        via ml_tabular.mlops.mlflow_utils.log_metrics.

    Returns
    -------
    history: Dict[str, Dict[str, float]]
        A dict with per-epoch summaries. Keys:
          - "train_<metric>" and "val_<metric>" if val_loader is provided.
          - The values are the metrics from the last epoch (for a full history,
            you can extend this to store lists per epoch in your training script).

        For simplicity, this implementation returns only the last-epoch metrics.
        You can adapt it to return the full history if desired.
    """
    if num_epochs <= 0:
        raise TrainingError(
            "num_epochs must be a positive integer.",
            code="fit_bad_num_epochs",
            context={"num_epochs": num_epochs},
            location="ml_tabular.torch.training.loops.fit",
        )

    model.to(device)
    history: Dict[str, float] = {}

    mlflow_enabled = use_mlflow and mlflow_is_enabled()

    for epoch in range(1, num_epochs + 1):
        logger.info("Epoch %d/%d", epoch, num_epochs)

        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            metrics=metrics,
            scaler=scaler,
            max_grad_norm=max_grad_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_interval=log_interval,
        )

        epoch_summary: Dict[str, float] = {}
        for name, value in train_metrics.items():
            key = f"train_{name}"
            epoch_summary[key] = float(value)

        logger.info(
            "Epoch %d train metrics: %s",
            epoch,
            {k: round(v, 6) for k, v in epoch_summary.items()},
        )

        val_metrics: Dict[str, float] = {}
        if val_loader is not None:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                loss_fn=loss_fn,
                device=device,
                metrics=metrics,
            )
            for name, value in val_metrics.items():
                key = f"val_{name}"
                epoch_summary[key] = float(value)

            logger.info(
                "Epoch %d val metrics: %s",
                epoch,
                {
                    k: round(v, 6)
                    for k, v in epoch_summary.items()
                    if k.startswith("val_")
                },
            )

        # Scheduler step
        if scheduler is not None:
            try:
                if scheduler_on == "epoch":
                    scheduler.step()
                elif scheduler_on == "val_metric":
                    if val_loader is None:
                        raise TrainingError(
                            "scheduler_on='val_metric' requires a validation loader.",
                            code="scheduler_val_metric_no_val_loader",
                            context={},
                            location="ml_tabular.torch.training.loops.fit",
                        )
                    if scheduler_metric not in val_metrics:
                        raise TrainingError(
                            f"scheduler_metric '{scheduler_metric}' not found in val_metrics.",
                            code="scheduler_metric_missing",
                            context={"available": list(val_metrics.keys())},
                            location="ml_tabular.torch.training.loops.fit",
                        )
                    scheduler.step(val_metrics[scheduler_metric])
                else:
                    raise TrainingError(
                        f"Unsupported scheduler_on value: {scheduler_on!r}.",
                        code="scheduler_bad_mode",
                        context={"scheduler_on": scheduler_on},
                        location="ml_tabular.torch.training.loops.fit",
                    )
            except Exception as exc:
                raise TrainingError(
                    "Error while stepping LR scheduler.",
                    code="scheduler_step_error",
                    cause=exc,
                    context={"scheduler_on": scheduler_on},
                    location="ml_tabular.torch.training.loops.fit",
                ) from exc

        # Early stopping
        if early_stopping is not None and val_loader is not None:
            early_stopping.step(val_metrics)
            if early_stopping.should_stop:
                logger.info(
                    "Early stopping triggered at epoch %d (best %s=%.6f).",
                    epoch,
                    early_stopping.monitor,
                    early_stopping.best_score,
                )
                history.update(epoch_summary)
                if mlflow_enabled:
                    mlflow_log_metrics(epoch_summary, step=epoch)
                break

        # MLflow logging
        if mlflow_enabled:
            mlflow_log_metrics(epoch_summary, step=epoch)

        history.update(epoch_summary)

    return {"last_epoch": history}
