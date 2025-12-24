# src/ecom_fraud/evaluation/metrics.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    make_scorer,
)


__all__ = [
    "BinaryMetrics",
    "ThresholdPoint",
    "evaluate_binary_at_threshold",
    "compute_pr_curve",
    "compute_roc_curve",
    "run_threshold_analysis",
    "summarize_thresholds",
    "metrics_dict_from_binary",
    "metrics_frame_from_models",
    "pr_auc_scorer",
]


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BinaryMetrics:
    """
    Comprehensive set of metrics for a binary classifier at a given threshold.

    This is intentionally richer than what you'd log in MLflow, because you
    also want it for analysis in notebooks and comparison tables.
    """

    threshold: float

    roc_auc: float
    pr_auc: float

    accuracy: float
    precision: float
    recall: float
    f1: float

    support_pos: int
    support_neg: int

    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        """
        Flatten into a dict for logging (e.g. MLflow). `prefix` can be used to
        distinguish train/test or model family (e.g. "train_", "test_").
        """
        p = f"{prefix}" if prefix in ("", "_") else f"{prefix}"
        return {
            f"{p}threshold": float(self.threshold),
            f"{p}roc_auc": float(self.roc_auc),
            f"{p}pr_auc": float(self.pr_auc),
            f"{p}accuracy": float(self.accuracy),
            f"{p}precision": float(self.precision),
            f"{p}recall": float(self.recall),
            f"{p}f1": float(self.f1),
            f"{p}support_pos": int(self.support_pos),
            f"{p}support_neg": int(self.support_neg),
            f"{p}tp": int(self.tp),
            f"{p}fp": int(self.fp),
            f"{p}tn": int(self.tn),
            f"{p}fn": int(self.fn),
        }


@dataclass
class ThresholdPoint:
    """
    A smaller view of performance at a particular operating threshold.

    This is what you typically care about when selecting operating points
    for product / business:
      - "best F1"
      - "high recall mode"
      - "high precision mode"
    """

    threshold: float
    precision: float
    recall: float
    f1: float
    support_pos: int
    support_neg: int
    tp: int
    fp: int
    tn: int
    fn: int

    @classmethod
    def from_binary_metrics(cls, m: BinaryMetrics) -> "ThresholdPoint":
        return cls(
            threshold=m.threshold,
            precision=m.precision,
            recall=m.recall,
            f1=m.f1,
            support_pos=m.support_pos,
            support_neg=m.support_neg,
            tp=m.tp,
            fp=m.fp,
            tn=m.tn,
            fn=m.fn,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(y: Any) -> np.ndarray:
    """Convert various array-likes to a flat numpy array."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        arr = arr.ravel()
    return arr


def _safe_metric(fn: Callable[..., float], *args: Any, **kwargs: Any) -> float:
    """
    Wrap a sklearn metric that might raise ValueError when only one class is
    present. Returns NaN in those edge cases instead of blowing up.
    """
    try:
        return float(fn(*args, **kwargs))
    except ValueError:
        return float("nan")


def _binary_confusion_counts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    pos_label: int = 1,
) -> Dict[str, int]:
    """
    Compute TP/FP/TN/FN counts manually. This avoids pulling in confusion_matrix
    and keeps us explicit.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)

    pos = pos_label
    neg = 1 - pos

    tp = int(((y_true == pos) & (y_pred == pos)).sum())
    fp = int(((y_true == neg) & (y_pred == pos)).sum())
    tn = int(((y_true == neg) & (y_pred == neg)).sum())
    fn = int(((y_true == pos) & (y_pred == neg)).sum())

    support_pos = int((y_true == pos).sum())
    support_neg = int((y_true == neg).sum())

    return dict(tp=tp, fp=fp, tn=tn, fn=fn, support_pos=support_pos, support_neg=support_neg)


# ---------------------------------------------------------------------------
# Public evaluation functions
# ---------------------------------------------------------------------------


def evaluate_binary_at_threshold(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    threshold: float = 0.5,
    pos_label: int = 1,
) -> BinaryMetrics:
    """
    Evaluate binary metrics for a classifier at a specific decision threshold.

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0/1).
    y_proba
        Predicted probabilities for the positive class (same shape as y_true).
    threshold
        Decision threshold in [0, 1].
    pos_label
        Label considered as the "positive" class (default: 1).

    Returns
    -------
    BinaryMetrics
        Dataclass with metrics & confusion counts.
    """
    y_true_arr = _to_numpy(y_true)
    y_proba_arr = _to_numpy(y_proba)

    if not (0.0 <= threshold <= 1.0):
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")

    y_pred = (y_proba_arr >= threshold).astype(int)

    # Core confusion counts
    counts = _binary_confusion_counts(y_true_arr, y_pred, pos_label=pos_label)

    # AUC metrics are threshold-independent; we always compute them from probabilities.
    roc_auc = _safe_metric(roc_auc_score, y_true_arr, y_proba_arr)
    pr_auc = _safe_metric(average_precision_score, y_true_arr, y_proba_arr)

    accuracy = _safe_metric(accuracy_score, y_true_arr, y_pred)
    precision = _safe_metric(precision_score, y_true_arr, y_pred, zero_division=0)
    recall = _safe_metric(recall_score, y_true_arr, y_pred, zero_division=0)
    f1_val = _safe_metric(f1_score, y_true_arr, y_pred, zero_division=0)

    return BinaryMetrics(
        threshold=float(threshold),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1_val,
        support_pos=counts["support_pos"],
        support_neg=counts["support_neg"],
        tp=counts["tp"],
        fp=counts["fp"],
        tn=counts["tn"],
        fn=counts["fn"],
    )


# ---------------------------------------------------------------------------
# Curves for plotting (PR / ROC)
# ---------------------------------------------------------------------------


def compute_pr_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
) -> pd.DataFrame:
    """
    Compute precision-recall curve points.

    Returns a DataFrame with columns:
        - threshold (len = n_thresholds)
        - precision
        - recall

    Note: sklearn's precision_recall_curve returns precision/recall of length
    (n_thresholds + 1). We align lengths by dropping the last precision/recall
    entry, so `threshold[i]` corresponds to (precision[i], recall[i]).
    """
    y_true_arr = _to_numpy(y_true)
    y_proba_arr = _to_numpy(y_proba)

    precision, recall, thresholds = precision_recall_curve(y_true_arr, y_proba_arr)

    if thresholds.size == 0:
        # Degenerate case: all probs identical; return a single row.
        return pd.DataFrame(
            {
                "threshold": [0.5],
                "precision": [float("nan")],
                "recall": [float("nan")],
            }
        )

    # Align lengths
    precision = precision[:-1]
    recall = recall[:-1]

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision,
            "recall": recall,
        }
    )


def compute_roc_curve(
    y_true: Sequence[int],
    y_proba: Sequence[float],
) -> pd.DataFrame:
    """
    Compute ROC curve points.

    Returns a DataFrame with columns:
        - fpr: false positive rate
        - tpr: true positive rate
        - threshold
    """
    y_true_arr = _to_numpy(y_true)
    y_proba_arr = _to_numpy(y_proba)

    try:
        fpr, tpr, thresholds = roc_curve(y_true_arr, y_proba_arr)
    except ValueError:
        # Degenerate case: only one class present.
        return pd.DataFrame(
            {
                "fpr": [0.0],
                "tpr": [0.0],
                "threshold": [0.5],
            }
        )

    return pd.DataFrame(
        {
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds,
        }
    )


# ---------------------------------------------------------------------------
# Threshold search / operating points
# ---------------------------------------------------------------------------


def _generate_candidate_thresholds(
    y_proba: np.ndarray,
    n_thresholds: int = 200,
) -> np.ndarray:
    """
    Generate a set of candidate thresholds based on probability quantiles.

    Using quantiles instead of all unique probabilities keeps this robust for
    large datasets (e.g. hundreds of thousands of rows).
    """
    y_proba = _to_numpy(y_proba)

    # Edge case: constant probabilities
    if np.allclose(y_proba, y_proba[0]):
        return np.array([0.5], dtype=float)

    qs = np.linspace(0.01, 0.99, n_thresholds)
    thresholds = np.quantile(y_proba, qs)

    # Ensure uniqueness and clip to [0, 1]
    thresholds = np.unique(np.clip(thresholds, 0.0, 1.0))

    # Common default threshold is always nice to include
    thresholds = np.unique(np.concatenate([thresholds, np.array([0.5])]))
    return thresholds


def run_threshold_analysis(
    y_true: Sequence[int],
    y_proba: Sequence[float],
    n_thresholds: int = 200,
    high_recall_target: float = 0.90,
    high_precision_target: float = 0.90,
    pos_label: int = 1,
) -> Dict[str, ThresholdPoint]:
    """
    Compute a small set of "named" operating points for a binary classifier.

    The idea here is to make your notebooks read like:

        thresholds = run_threshold_analysis(y_test, y_test_proba)
        thresholds["best_f1"]
        thresholds["high_recall"]
        thresholds["high_precision"]

    which makes it obvious to reviewers that you think in terms of trade-offs,
    not just a single metric.

    Named thresholds:
      - best_f1:   threshold with highest F1 on the grid.
      - high_recall:    highest threshold achieving recall >= high_recall_target.
      - high_precision: lowest threshold achieving precision >= high_precision_target.

    Returns
    -------
    Dict[str, ThresholdPoint]
        Mapping from a short name to a ThresholdPoint dataclass.
    """
    y_true_arr = _to_numpy(y_true)
    y_proba_arr = _to_numpy(y_proba)

    candidates = _generate_candidate_thresholds(y_proba_arr, n_thresholds=n_thresholds)

    metrics_list: list[BinaryMetrics] = []
    for thr in candidates:
        m = evaluate_binary_at_threshold(
            y_true=y_true_arr,
            y_proba=y_proba_arr,
            threshold=float(thr),
            pos_label=pos_label,
        )
        metrics_list.append(m)

    # best_f1: straightforward argmax
    best_f1_metrics = max(metrics_list, key=lambda m: (m.f1, m.threshold))
    best_f1 = ThresholdPoint.from_binary_metrics(best_f1_metrics)

    # high_recall: we want threshold with recall >= target, but as large as possible
    high_recall_candidates = [m for m in metrics_list if m.recall >= high_recall_target]
    if high_recall_candidates:
        high_recall_metrics = max(high_recall_candidates, key=lambda m: m.threshold)
    else:
        # fallback: just the best recall on the grid
        high_recall_metrics = max(metrics_list, key=lambda m: (m.recall, m.threshold))
    high_recall = ThresholdPoint.from_binary_metrics(high_recall_metrics)

    # high_precision: we want threshold with precision >= target, but as small as possible
    high_precision_candidates = [m for m in metrics_list if m.precision >= high_precision_target]
    if high_precision_candidates:
        high_precision_metrics = min(high_precision_candidates, key=lambda m: m.threshold)
    else:
        # fallback: best precision on grid
        high_precision_metrics = max(metrics_list, key=lambda m: (m.precision, -m.threshold))
    high_precision = ThresholdPoint.from_binary_metrics(high_precision_metrics)

    return {
        "best_f1": best_f1,
        "high_recall": high_recall,
        "high_precision": high_precision,
    }


def summarize_thresholds(
    thresholds: Mapping[str, ThresholdPoint],
) -> pd.DataFrame:
    """
    Turn a threshold dict into a tidy DataFrame for inspection / plotting.

    Columns:
      - name
      - threshold
      - precision
      - recall
      - f1
      - support_pos
      - support_neg
      - tp, fp, tn, fn
    """
    rows: list[Dict[str, Any]] = []
    for name, tp in thresholds.items():
        rows.append(
            {
                "name": name,
                "threshold": tp.threshold,
                "precision": tp.precision,
                "recall": tp.recall,
                "f1": tp.f1,
                "support_pos": tp.support_pos,
                "support_neg": tp.support_neg,
                "tp": tp.tp,
                "fp": tp.fp,
                "tn": tp.tn,
                "fn": tp.fn,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Utilities for MLflow and comparison tables
# ---------------------------------------------------------------------------


def metrics_dict_from_binary(
    metrics: BinaryMetrics,
    prefix: str = "",
) -> Dict[str, float]:
    """
    Convenience wrapper for logging a BinaryMetrics object into MLflow or
    another tracking system.

    Example:
        log_metrics(metrics_dict_from_binary(test_metrics, prefix="test_"))
    """
    # Normalize prefix: "" or something like "test_"
    if prefix and not prefix.endswith("_"):
        prefix = prefix + "_"
    return metrics.as_dict(prefix=prefix)


def metrics_frame_from_models(
    model_metrics: Mapping[str, BinaryMetrics],
) -> pd.DataFrame:
    """
    Build a comparison table across models at their *chosen* thresholds.

    Parameters
    ----------
    model_metrics
        Mapping like:
            {
                "xgb_tuned": BinaryMetrics(...),
                "lgbm_tuned": BinaryMetrics(...),
                "mlp_hybrid": BinaryMetrics(...),
            }

    Returns
    -------
    DataFrame
        One row per model with columns:
            model, threshold, roc_auc, pr_auc, accuracy, precision, recall, f1,
            support_pos, support_neg, tp, fp, tn, fn
    """
    rows: list[Dict[str, Any]] = []
    for name, m in model_metrics.items():
        row = {"model": name}
        row.update(m.as_dict(prefix=""))
        rows.append(row)
    df = pd.DataFrame(rows)
    # Put model column first
    cols = ["model"] + [c for c in df.columns if c != "model"]
    return df[cols]


# ---------------------------------------------------------------------------
# PR-AUC scorer for sklearn model_selection
# ---------------------------------------------------------------------------


def pr_auc_scorer(estimator: Any, X: Any, y_true: Sequence[int]) -> float:
    """
    Custom scorer for use with sklearn's GridSearchCV / RandomizedSearchCV.

    Assumes the estimator implements predict_proba(X) and that the positive
    class is at index 1.

    Usage:
        from sklearn.model_selection import RandomizedSearchCV

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=...,
            scoring=pr_auc_scorer,
            ...
        )
    """
    y_true_arr = _to_numpy(y_true)
    proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y_true_arr, proba)


# If you also want the `make_scorer` flavor pre-built, you can expose this:
pr_auc_sklearn_scorer = make_scorer(average_precision_score, needs_proba=True)
