from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, MutableMapping, Optional, Tuple

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from ecom_fraud.exceptions import DataError, TrainingError
from ecom_fraud.logging_config import get_logger
from ecom_fraud.mlops.mlflow_utils import (
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
        "max" -> higher is better (e.g., accuracy/AUC).
    patience:
        Number of epochs with no improvement after which training is stopped.
    min_delta:
        Minimum absolute change to qualify as an improvement.

    Attributes
    ----------
    best_score:
        Best value of the monitored metric seen so far.
    best_epoch:
        Epoch index (1-based) at which best_score was achieved, if known.
    num_bad_epochs:
        Number of consecutive epochs with no improvement.
    should_stop:
        Flag indicating that early stopping should trigger.
    """

    monitor: str = "val_loss"
    mode: str = "min"  # "min" or "max"
    patience: int = 10
    min_delta: float = 0.0

    best_score: Optional[float] = None
    best_epoch: Optional[int] = None
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

    def step(self, metrics: Mapping[str, float], *, epoch: Optional[int] = None) -> None:
        """Update the early-stopping state with metrics from the latest epoch.

        Parameters
        ----------
        metrics:
            Mapping from metric name to value (e.g. {"loss": 0.23, "pr_auc": 0.81}).
        epoch:
            Optional epoch index (1-based) associated with this metrics dict.
            If provided, best_epoch will be updated whenever best_score improves.
        """
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
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            logger.info(
                "EarlyStopping init: monitor=%s, best_score=%.6f, best_epoch=%s",
                self.monitor,
                self.best_score,
                str(self.best_epoch),
            )
            return

        if self._is_improvement(current, self.best_score):
            self.best_score = current
            self.best_epoch = epoch
            self.num_bad_epochs = 0
            logger.info(
                "EarlyStopping: improvement detected on %s, new best_score=%.6f at epoch=%s",
                self.monitor,
                self.best_score,
                str(self.best_epoch),
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


def _to_device(obj: Any, device: torch.device) -> Any:
    """Recursively move tensors in a nested structure to the specified device.

    Supported structures:
      - torch.Tensor
      - mappings (e.g. dict[str, Any])
      - sequences (list/tuple) of the above

    Non-tensor leaf values are returned unchanged.
    """
    if isinstance(obj, torch.Tensor):
        return obj.to(device)

    # Mapping: move each value
    if isinstance(obj, Mapping):
        return {k: _to_device(v, device) for k, v in obj.items()}

    # Sequence: preserve type (list/tuple)
    if isinstance(obj, (list, tuple)):
        seq_type = type(obj)
        return seq_type(_to_device(v, device) for v in obj)

    # Anything else: leave as-is
    return obj


def _infer_batch_size_from_inputs(inputs: Any) -> int:
    """Infer batch size from (possibly nested) inputs.

    The function looks for the first tensor-like leaf and returns its
    leading dimension as the batch size.

    Supported structures:
      - torch.Tensor  -> uses inputs.shape[0]
      - mappings      -> inspects values
      - sequences     -> inspects elements

    Raises
    ------
    TrainingError
        If no suitable tensor is found or the tensor is scalar.
    """
    if isinstance(inputs, torch.Tensor):
        if inputs.ndim == 0:
            raise TrainingError(
                "Cannot infer batch size from a scalar tensor.",
                code="batch_size_infer_scalar",
                context={"shape": tuple(inputs.shape)},
                location="ml_tabular.torch.training.loops._infer_batch_size_from_inputs",
            )
        return int(inputs.shape[0])

    if isinstance(inputs, Mapping):
        for value in inputs.values():
            if isinstance(value, (torch.Tensor, Mapping, list, tuple)):
                return _infer_batch_size_from_inputs(value)
        raise TrainingError(
            "Could not infer batch size from mapping inputs: no tensor-like values found.",
            code="batch_size_infer_mapping_failed",
            context={"type": type(inputs).__name__},
            location="ml_tabular.torch.training.loops._infer_batch_size_from_inputs",
        )

    if isinstance(inputs, (list, tuple)):
        for value in inputs:
            if isinstance(value, (torch.Tensor, Mapping, list, tuple)):
                return _infer_batch_size_from_inputs(value)
        raise TrainingError(
            "Could not infer batch size from sequence inputs: no tensor-like elements found.",
            code="batch_size_infer_sequence_failed",
            context={"type": type(inputs).__name__},
            location="ml_tabular.torch.training.loops._infer_batch_size_from_inputs",
        )

    raise TrainingError(
        "Could not infer batch size from inputs; expected a tensor, sequence of "
        "tensors, or mapping of tensors.",
        code="batch_size_infer_failed",
        context={"type": type(inputs).__name__},
        location="ml_tabular.torch.training.loops._infer_batch_size_from_inputs",
    )


def _move_batch_to_device(
    batch: Any,
    device: torch.device,
) -> Tuple[Any, Optional[Any]]:
    """Move a batch from a DataLoader to the specified device.

    Expects:
      - batch to be (inputs, targets), or
      - batch to be inputs only (targets=None).

    The `inputs` object may be:
      - a torch.Tensor,
      - a tuple/list of tensors,
      - a dict mapping keys to tensors (or nested structures of those).

    All tensors in both inputs and targets are recursively moved to `device`.

    Returns
    -------
    (inputs, targets)
        Both on the specified device; targets may be None for inference-only
        loops. The structure of `inputs` is preserved.
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

    x = _to_device(x, device)
    if y is not None:
        y = _to_device(y, device)

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

    NOTE: this helper is kept for backwards compatibility but is no longer
    used in the main train/eval loops, which now compute metrics globally.
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
    """Compute average loss and metrics over an epoch.

    NOTE: kept for compatibility; main loops now handle metrics directly.
    """
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
    max_grad_norm: Optional[float] = 1.0,
    gradient_accumulation_steps: int = 1,
    log_interval: Optional[int] = None,
) -> Dict[str, float]:
    """Run a single training epoch.

    Loss is averaged over all samples in the epoch.

    If metrics are provided, each metric is computed ONCE per epoch on the
    concatenated predictions and targets (global metrics), which is correct
    for AUC / PR-AUC / F1 / etc.

    Parameters
    ----------
    model, dataloader, optimizer, loss_fn, device:
        Standard PyTorch training components.
    metrics:
        Optional mapping of metric name -> callable(y_true, y_pred_logits).
    scaler:
        Optional GradScaler for mixed-precision (AMP) training.
    max_grad_norm:
        If not None, gradients are clipped to this norm before each optimizer
        step. Defaults to 1.0, which is a safe general-purpose choice.
    gradient_accumulation_steps:
        Accumulate gradients over this many batches before stepping the
        optimizer (useful for large batch sizes).
    log_interval:
        Optional batch-level logging interval.
    """
    model.train()

    metric_fns = metrics or {}
    loss_sum = 0.0
    n_samples = 0

    # For global metrics, accumulate all targets and predictions
    all_y_true = [] if metric_fns else None
    all_y_pred = [] if metric_fns else None

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

        batch_size = _infer_batch_size_from_inputs(x)

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

        # Backward pass (with optional AMP)
        if use_amp:
            assert scaler is not None
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update loss accumulators (undo division by gradient_accumulation_steps)
        loss_sum += float(loss.detach().cpu().item()) * batch_size * gradient_accumulation_steps
        n_samples += batch_size

        # Collect predictions/targets for global metrics
        if metric_fns:
            all_y_true.append(y.detach().cpu())
            all_y_pred.append(preds.detach().cpu())

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

    if n_samples <= 0:
        raise TrainingError(
            "No samples were seen during the training epoch.",
            code="no_samples_seen_train",
            context={},
            location="ml_tabular.torch.training.loops.train_one_epoch",
        )

    results: Dict[str, float] = {}
    results["loss"] = loss_sum / n_samples

    # Compute global metrics once on the full epoch
    if metric_fns:
        y_true_full = torch.cat(all_y_true, dim=0)  # type: ignore[arg-type]
        y_pred_full = torch.cat(all_y_pred, dim=0)  # type: ignore[arg-type]

        for name, fn in metric_fns.items():
            try:
                value = float(fn(y_true_full, y_pred_full))
            except Exception as exc:
                raise TrainingError(
                    f"Error while computing metric '{name}' on full training epoch.",
                    code="metric_computation_error_train",
                    cause=exc,
                    context={"metric_name": name},
                    location="ml_tabular.torch.training.loops.train_one_epoch",
                ) from exc
            results[name] = value

    return results


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
    *,
    metrics: Optional[Mapping[str, MetricFn]] = None,
) -> Dict[str, float]:
    """Evaluate a model on a validation/test DataLoader.

    Loss is averaged over all samples.

    If metrics are provided, each metric is computed ONCE per evaluation on
    the concatenated predictions and targets (global metrics).
    """
    model.eval()

    metric_fns = metrics or {}
    loss_sum = 0.0
    n_samples = 0

    all_y_true = [] if metric_fns else None
    all_y_pred = [] if metric_fns else None

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

            batch_size = _infer_batch_size_from_inputs(x)
            loss_sum += float(loss.detach().cpu().item()) * batch_size
            n_samples += batch_size

            if metric_fns:
                all_y_true.append(y.detach().cpu())
                all_y_pred.append(preds.detach().cpu())

    if n_samples <= 0:
        raise TrainingError(
            "No samples were seen during evaluation.",
            code="no_samples_seen_eval",
            context={},
            location="ml_tabular.torch.training.loops.evaluate",
        )

    results: Dict[str, float] = {}
    results["loss"] = loss_sum / n_samples

    if metric_fns:
        y_true_full = torch.cat(all_y_true, dim=0)  # type: ignore[arg-type]
        y_pred_full = torch.cat(all_y_pred, dim=0)  # type: ignore[arg-type]

        for name, fn in metric_fns.items():
            try:
                value = float(fn(y_true_full, y_pred_full))
            except Exception as exc:
                raise TrainingError(
                    f"Error while computing metric '{name}' on full eval set.",
                    code="metric_computation_error_eval",
                    cause=exc,
                    context={"metric_name": name},
                    location="ml_tabular.torch.training.loops.evaluate",
                ) from exc
            results[name] = value

    return results


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
    max_grad_norm: Optional[float] = 1.0,
    gradient_accumulation_steps: int = 1,
    log_interval: Optional[int] = None,
    early_stopping: Optional[EarlyStopping] = None,
    use_mlflow: bool = True,
) -> Dict[str, Any]:
    """Train a model for multiple epochs with optional validation, scheduler, and MLflow.

    Parameters
    ----------
    model, train_loader, optimizer, loss_fn, device:
        Standard training components.
    num_epochs:
        Number of epochs to train for.
    metrics:
        Optional mapping of metric name -> callable(y_true, y_pred) for both
        training and validation. These are computed globally per epoch.
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
        Optional gradient clipping norm. Defaults to 1.0 for safer training.
    gradient_accumulation_steps:
        Accumulate gradients over this many batches in train_one_epoch.
    log_interval:
        Passed through to train_one_epoch for batch-level logging.
    early_stopping:
        Optional EarlyStopping instance. If provided and should_stop becomes True,
        training stops early. When used with a validation loader, the returned
        summary will include a "best_epoch" entry.
    use_mlflow:
        If True and MLflow is enabled in config, per-epoch metrics are logged.

    Returns
    -------
    summary: Dict[str, Any]
        Keys:
          - "last_epoch": Dict[str, float]
                Metrics from the final epoch (for backward compatibility with
                previous versions that returned only this dict).
          - "history": List[Dict[str, float]]
                Per-epoch summaries. Each dict contains train_*/val_* metrics
                and an "epoch" field.
          - "best_epoch": Dict[str, float] (optional)
                Summary dict from the epoch with the best monitored metric
                (when EarlyStopping with a validation loader is used).
    """
    if num_epochs <= 0:
        raise TrainingError(
            "num_epochs must be a positive integer.",
            code="fit_bad_num_epochs",
            context={"num_epochs": num_epochs},
            location="ml_tabular.torch.training.loops.fit",
        )

    model.to(device)

    # Full per-epoch history
    epoch_history: list[Dict[str, float]] = []
    last_epoch_summary: Dict[str, float] = {}

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

        # Early stopping (only meaningful if we have validation metrics)
        if early_stopping is not None and val_loader is not None:
            early_stopping.step(val_metrics, epoch=epoch)

        # Add epoch index to summary (stored as float for consistency with metrics)
        epoch_summary["epoch"] = float(epoch)

        # MLflow logging (log all keys, including epoch)
        if mlflow_enabled:
            mlflow_log_metrics(epoch_summary, step=epoch)

        # Update running history
        last_epoch_summary = epoch_summary
        epoch_history.append(epoch_summary)

        # Check early stopping *after* logging and history update
        if early_stopping is not None and early_stopping.should_stop:
            logger.info(
                "Early stopping triggered at epoch %d (best %s=%.6f at epoch=%s).",
                epoch,
                early_stopping.monitor,
                early_stopping.best_score if early_stopping.best_score is not None else float("nan"),
                str(early_stopping.best_epoch),
            )
            break

    # Determine best_epoch summary if early stopping was used with a validation loader
    best_epoch_summary: Optional[Dict[str, float]] = None
    if early_stopping is not None and early_stopping.best_epoch is not None:
        target_epoch = int(early_stopping.best_epoch)
        for ep_summary in epoch_history:
            ep_idx = int(ep_summary.get("epoch", -1))
            if ep_idx == target_epoch:
                best_epoch_summary = ep_summary
                break

    result: Dict[str, Any] = {
        "last_epoch": last_epoch_summary,
        "history": epoch_history,
    }
    if best_epoch_summary is not None:
        result["best_epoch"] = best_epoch_summary

    return result
