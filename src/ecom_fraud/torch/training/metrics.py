from __future__ import annotations

from typing import Callable, Dict

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from ecom_fraud.exceptions import TrainingError
from ecom_fraud.logging_config import get_logger

logger = get_logger(__name__)

MetricFn = Callable[[torch.Tensor, torch.Tensor], float]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy_1d(
    tensor: torch.Tensor,
    *,
    name: str,
    location: str,
) -> np.ndarray:
    """Convert a tensor to a contiguous 1D numpy array.

    This is robust to device/dtype and enforces a flat shape, which is what
    sklearn expects.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TrainingError(
            f"{name} must be a torch.Tensor, got {type(tensor).__name__}",
            code="metric_bad_type",
            context={"name": name, "type": type(tensor).__name__},
            location=location,
        )

    t = tensor.detach().cpu().reshape(-1)
    return t.numpy()


def _prepare_binary_logits_inputs(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    *,
    location: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and normalize inputs for binary metrics on logits.

    Assumptions
    -----------
    - y_true contains binary targets {0, 1} (ints or floats).
    - y_pred_logits are raw logits (real-valued), NOT probabilities.
    """
    y_true_np = _to_numpy_1d(y_true, name="y_true", location=location)
    y_pred_np = _to_numpy_1d(y_pred_logits, name="y_pred_logits", location=location)

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise TrainingError(
            "y_true and y_pred_logits must have the same number of samples.",
            code="metric_shape_mismatch",
            context={
                "n_true": int(y_true_np.shape[0]),
                "n_pred": int(y_pred_np.shape[0]),
            },
            location=location,
        )

    # Ensure binary {0, 1} labels (allow 0.0/1.0 floats)
    unique = np.unique(y_true_np)
    if not np.isin(unique, [0.0, 1.0]).all():
        raise TrainingError(
            "Binary metrics expect y_true to contain only {0, 1}.",
            code="metric_non_binary_targets",
            context={"unique_values": unique.tolist()},
            location=location,
        )

    # Convert logits -> probabilities via sigmoid
    # Clamp to avoid numerical overflow for very large magnitudes.
    logits_clamped = np.clip(y_pred_np, -20.0, 20.0)
    y_prob_np = 1.0 / (1.0 + np.exp(-logits_clamped))

    return y_true_np, y_prob_np


def _safe_roc_auc(
    y_true_np: np.ndarray,
    y_prob_np: np.ndarray,
    *,
    location: str,
) -> float:
    """Compute ROC AUC with graceful handling of edge cases.

    If roc_auc_score raises ValueError (e.g. only one class present),
    we log a warning and return NaN rather than crashing the whole run.
    """
    try:
        return float(roc_auc_score(y_true_np, y_prob_np))
    except ValueError as exc:
        logger.warning(
            "roc_auc_score failed in %s with ValueError: %s. "
            "Returning NaN (check that both classes are present in y_true).",
            location,
            exc,
        )
        return float("nan")


def _safe_pr_auc(
    y_true_np: np.ndarray,
    y_prob_np: np.ndarray,
    *,
    location: str,
) -> float:
    """Compute PR AUC (average precision) with graceful error handling."""
    try:
        return float(average_precision_score(y_true_np, y_prob_np))
    except ValueError as exc:
        logger.warning(
            "average_precision_score failed in %s with ValueError: %s. "
            "Returning NaN.",
            location,
            exc,
        )
        return float("nan")


def _threshold_predictions(
    y_prob_np: np.ndarray,
    *,
    threshold: float,
    location: str,
) -> np.ndarray:
    """Convert probabilities to hard labels via a scalar threshold."""
    if not (0.0 < threshold < 1.0):
        raise TrainingError(
            f"Threshold must be in (0, 1), got {threshold}",
            code="metric_bad_threshold",
            context={"threshold": threshold},
            location=location,
        )
    return (y_prob_np >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Public metric builders
# ---------------------------------------------------------------------------


def build_binary_classification_metrics_from_logits(
    *,
    threshold: float = 0.5,
) -> Dict[str, MetricFn]:
    """Build a standard set of binary classification metrics for logits.

    The returned dict is designed to plug directly into the training loops
    in `torch/training/loops.py`, which expect metric functions of signature:

        fn(y_true: torch.Tensor, y_pred: torch.Tensor) -> float

    Here:

      * y_true is a binary tensor with values {0, 1}.
      * y_pred is a tensor of raw logits (unnormalized scores).
      * We apply sigmoid internally to obtain probabilities.

    Metrics returned
    ----------------
    - "auc":     ROC AUC, using sklearn.metrics.roc_auc_score.
    - "pr_auc":  Precision-Recall AUC, using average_precision_score.
    - "f1":      F1-score at the given threshold.
    - "precision": Precision at the given threshold.
    - "recall":    Recall at the given threshold.

    Edge cases
    ----------
    If AUC / PR-AUC cannot be computed (e.g. only one class present in y_true),
    we log a warning and return NaN rather than raising, to avoid crashing long
    training or hyperparameter search runs.

    Parameters
    ----------
    threshold:
        Scalar threshold in (0, 1) used to derive hard predictions for
        precision / recall / F1.

    Returns
    -------
    Dict[str, MetricFn]
        A mapping from metric name to callable.
    """
    location_base = "ecom_fraud.torch.training.metrics.binary_from_logits"
    if not (0.0 < threshold < 1.0):
        raise TrainingError(
            f"Threshold must be in (0, 1), got {threshold}",
            code="metric_bad_threshold",
            context={"threshold": threshold},
            location=f"{location_base}.builder",
        )

    def auc_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
        y_true_np, y_prob_np = _prepare_binary_logits_inputs(
            y_true,
            y_pred_logits,
            location=f"{location_base}.auc",
        )
        return _safe_roc_auc(
            y_true_np,
            y_prob_np,
            location=f"{location_base}.auc",
        )

    def pr_auc_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
        y_true_np, y_prob_np = _prepare_binary_logits_inputs(
            y_true,
            y_pred_logits,
            location=f"{location_base}.pr_auc",
        )
        return _safe_pr_auc(
            y_true_np,
            y_prob_np,
            location=f"{location_base}.pr_auc",
        )

    def precision_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
        y_true_np, y_prob_np = _prepare_binary_logits_inputs(
            y_true,
            y_pred_logits,
            location=f"{location_base}.precision",
        )
        y_pred_labels = _threshold_predictions(
            y_prob_np,
            threshold=threshold,
            location=f"{location_base}.precision",
        )
        precision, _, _, _ = precision_recall_fscore_support(
            y_true_np,
            y_pred_labels,
            average="binary",
            zero_division=0,
        )
        return float(precision)

    def recall_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
        y_true_np, y_prob_np = _prepare_binary_logits_inputs(
            y_true,
            y_pred_logits,
            location=f"{location_base}.recall",
        )
        y_pred_labels = _threshold_predictions(
            y_prob_np,
            threshold=threshold,
            location=f"{location_base}.recall",
        )
        _, recall, _, _ = precision_recall_fscore_support(
            y_true_np,
            y_pred_labels,
            average="binary",
            zero_division=0,
        )
        return float(recall)

    def f1_fn(y_true: torch.Tensor, y_pred_logits: torch.Tensor) -> float:
        y_true_np, y_prob_np = _prepare_binary_logits_inputs(
            y_true,
            y_pred_logits,
            location=f"{location_base}.f1",
        )
        y_pred_labels = _threshold_predictions(
            y_prob_np,
            threshold=threshold,
            location=f"{location_base}.f1",
        )
        _, _, f1, _ = precision_recall_fscore_support(
            y_true_np,
            y_pred_labels,
            average="binary",
            zero_division=0,
        )
        return float(f1)

    return {
        "auc": auc_fn,
        "pr_auc": pr_auc_fn,
        "precision": precision_fn,
        "recall": recall_fn,
        "f1": f1_fn,
    }


def build_basic_binary_classification_metrics_from_logits(
    *,
    threshold: float = 0.5,
) -> Dict[str, MetricFn]:
    """Build a slimmer metric set (AUC + PR-AUC) for quicker experiments.

    This is a thin wrapper around `build_binary_classification_metrics_from_logits`,
    returning only the two most common metrics used for model selection:

      - "auc"
      - "pr_auc"

    Parameters
    ----------
    threshold:
        Passed through to the underlying builder (even though it's only
        used for threshold-based metrics). This keeps the signature
        consistent across metric builders.

    Returns
    -------
    Dict[str, MetricFn]
        A mapping containing only "auc" and "pr_auc".
    """
    full = build_binary_classification_metrics_from_logits(threshold=threshold)
    return {k: full[k] for k in ("auc", "pr_auc")}


__all__ = [
    "MetricFn",
    "build_binary_classification_metrics_from_logits",
    "build_basic_binary_classification_metrics_from_logits",
]
