# src/ecom_fraud/evaluation/plots.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
)

__all__ = [
    "plot_precision_recall_curve",
    "plot_roc_curve",
    "plot_threshold_curves",
    "plot_training_curves",
    "plot_model_metric_bar",
    "plot_multi_precision_recall",
    "plot_multi_roc",
    "plot_metric_over_epochs_multi",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_ax(ax: Optional[plt.Axes] = None, figsize: Tuple[int, int] = (8, 6)) -> plt.Axes:
    """
    Return an existing Axes or create a new one with a default figsize.
    """
    if ax is not None:
        return ax

    fig, new_ax = plt.subplots(figsize=figsize)
    return new_ax


def _maybe_save(fig: plt.Figure, savepath: Optional[Path | str], dpi: int = 120) -> None:
    """
    Save figure to disk if savepath is provided.
    """
    if savepath is None:
        return

    savepath = Path(savepath)
    savepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(savepath, dpi=dpi, bbox_inches="tight")


# ---------------------------------------------------------------------------
# 1. PR & ROC curves
# ---------------------------------------------------------------------------


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot a precision–recall curve for a single model.

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0/1).
    y_proba
        Predicted probabilities for the positive class.
    label
        Optional label for the model in the legend. If not provided, we include
        the Average Precision (PR-AUC) in a default label.
    ax
        Optional matplotlib Axes. If None, a new figure is created.
    figsize
        Figure size if ax is None.
    savepath
        Optional path to save the figure to disk.
    show
        If True, call plt.show() at the end.

    Returns
    -------
    ax
        Matplotlib Axes containing the PR curve.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap = average_precision_score(y_true, y_proba)

    ax = _ensure_ax(ax, figsize=figsize)
    ax.plot(recall, precision, marker="", linewidth=2)

    if label is None:
        label = f"AP = {ap:.3f}"
    else:
        label = f"{label} (AP = {ap:.3f})"

    ax.set_title("Precision–Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend([label])
    ax.grid(True, alpha=0.3)

    fig = ax.get_figure()
    _maybe_save(fig, savepath)

    if show:
        plt.show()

    return ax


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    label: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot a ROC curve for a single model.

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0/1).
    y_proba
        Predicted probabilities for the positive class.
    label
        Optional label for the model. If None, we include ROC-AUC in label.
    ax
        Optional matplotlib Axes. If None, a new figure is created.
    figsize
        Figure size if ax is None.
    savepath
        Optional path to save the figure.
    show
        If True, call plt.show().

    Returns
    -------
    ax
        Matplotlib Axes with ROC curve plotted.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)

    ax = _ensure_ax(ax, figsize=figsize)
    ax.plot(fpr, tpr, linewidth=2)

    if label is None:
        label = f"AUC = {roc_auc:.3f}"
    else:
        label = f"{label} (AUC = {roc_auc:.3f})"

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend([label])
    ax.grid(True, alpha=0.3)

    fig = ax.get_figure()
    _maybe_save(fig, savepath)

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# 2. Precision/recall vs threshold curves
# ---------------------------------------------------------------------------


def plot_threshold_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot precision and recall as a function of the decision threshold.

    This is useful to choose business operating points (e.g. high-recall mode,
    high-precision mode) for fraud detection.

    Parameters
    ----------
    y_true
        Ground-truth binary labels.
    y_proba
        Predicted probabilities for the positive class.
    ax
        Optional Axes to draw on. If None, creates a new figure.
    figsize
        Figure size if ax is None.
    savepath
        Optional path to save the figure to disk.
    show
        If True, call plt.show().

    Returns
    -------
    ax
        Axes with precision & recall vs threshold curves.
    """
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    thresholds = thresholds if thresholds.size > 0 else np.array([0.5])

    # precision/recall curves have one extra point at the beginning;
    # align them to thresholds length for plotting vs threshold.
    precision_for_thr = precision[:-1]
    recall_for_thr = recall[:-1]

    ax = _ensure_ax(ax, figsize=figsize)

    ax.plot(thresholds, precision_for_thr, label="Precision", linewidth=2)
    ax.plot(thresholds, recall_for_thr, label="Recall", linewidth=2)

    ax.set_title("Precision & Recall vs Threshold")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = ax.get_figure()
    _maybe_save(fig, savepath)

    if show:
        plt.show()

    return ax


# ---------------------------------------------------------------------------
# 3. Training curves for MLP / hybrids
# ---------------------------------------------------------------------------


def plot_training_curves(
    history: Mapping[str, Sequence[float]],
    *,
    title_prefix: str = "Model",
    ax_loss: Optional[plt.Axes] = None,
    ax_metric1: Optional[plt.Axes] = None,
    ax_metric2: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (16, 4),
    metric1_key: str = "val_pr_auc",
    metric1_label: str = "Validation PR-AUC",
    metric2_key: str = "val_recall",
    metric2_label: str = "Validation Recall (thr=0.5)",
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> Dict[str, plt.Axes]:
    """
    Plot standard training curves from a history dict of the form you’re already using.

    Expected keys (where present):
      - "epoch": list of epoch numbers (1-based or 0-based)
      - "train_loss": list of train losses
      - "val_loss": list of validation losses (optional)
      - metric1_key: e.g. "val_pr_auc"
      - metric2_key: e.g. "val_recall"

    The function is tolerant of missing keys: it will only plot what’s available.

    This is designed to work directly with:
      - `history_lgbm` and `history_xgb` from your hybrid notebooks
      - the `history` returned by your `train_hybrid_for_curves` helper

    Parameters
    ----------
    history
        Dict-like with lists of per-epoch values.
    title_prefix
        String prefix used in subplot titles (e.g. "LGBM-hybrid MLP").
    ax_loss
        Optional Axes for loss curves (train vs val).
    ax_metric1
        Optional Axes for primary validation metric (e.g. PR-AUC).
    ax_metric2
        Optional Axes for secondary validation metric (e.g. recall).
    figsize
        Figure size if we create a new figure with 3 panels.
    metric1_key, metric1_label
        Key and label for the primary validation metric.
    metric2_key, metric2_label
        Key and label for the secondary validation metric.
    savepath
        Optional path to save the entire figure.
    show
        If True, call plt.show().

    Returns
    -------
    axes
        Dict with entries: {"loss": ax_loss, "metric1": ax_metric1, "metric2": ax_metric2}
    """
    epochs = list(history.get("epoch", []))
    if not epochs:
        # fallback: infer epochs from length of train_loss
        n = len(history.get("train_loss", []))
        epochs = list(range(1, n + 1))

    # Create a 1x3 figure if no axes provided
    if any(ax is None for ax in (ax_loss, ax_metric1, ax_metric2)):
        fig, (ax_loss, ax_metric1, ax_metric2) = plt.subplots(
            1, 3, figsize=figsize
        )
    else:
        fig = ax_loss.get_figure()

    # ------------------------------
    # Loss (train vs val)
    # ------------------------------
    train_loss = history.get("train_loss")
    val_loss = history.get("val_loss")

    if train_loss is not None:
        ax_loss.plot(epochs, train_loss, marker="o", label="Train loss")
    if val_loss is not None:
        ax_loss.plot(epochs, val_loss, marker="o", label="Val loss")

    ax_loss.set_title(f"{title_prefix}: Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True, alpha=0.3)
    if train_loss is not None or val_loss is not None:
        ax_loss.legend()

    # ------------------------------
    # Metric 1 (e.g., PR-AUC)
    # ------------------------------
    metric1_vals = history.get(metric1_key)
    if metric1_vals is not None:
        ax_metric1.plot(epochs, metric1_vals, marker="o")
        ax_metric1.set_title(f"{title_prefix}: {metric1_label}")
        ax_metric1.set_xlabel("Epoch")
        ax_metric1.set_ylabel(metric1_label)
        ax_metric1.grid(True, alpha=0.3)
    else:
        ax_metric1.set_visible(False)

    # ------------------------------
    # Metric 2 (e.g., recall)
    # ------------------------------
    metric2_vals = history.get(metric2_key)
    if metric2_vals is not None:
        ax_metric2.plot(epochs, metric2_vals, marker="o")
        ax_metric2.set_title(f"{title_prefix}: {metric2_label}")
        ax_metric2.set_xlabel("Epoch")
        ax_metric2.set_ylabel(metric2_label)
        ax_metric2.grid(True, alpha=0.3)
    else:
        ax_metric2.set_visible(False)

    plt.tight_layout()
    _maybe_save(fig, savepath)

    if show:
        plt.show()

    return {
        "loss": ax_loss,
        "metric1": ax_metric1,
        "metric2": ax_metric2,
    }


# ---------------------------------------------------------------------------
# 4. Model comparison bar plots
# ---------------------------------------------------------------------------


def plot_model_metric_bar(
    model_to_metric: Mapping[str, float],
    *,
    metric_name: str = "PR-AUC",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot a simple bar chart comparing models on a single scalar metric.

    This is handy for your final "XGB vs LGBM vs Hybrid MLP" comparison in the
    conclusion notebook.

    Parameters
    ----------
    model_to_metric
        Mapping from model name (e.g. "XGB tree-only", "LGBM-hybrid MLP") to
        a scalar metric value (e.g. PR-AUC on test).
    metric_name
        Name of the metric to display on the y-axis and title.
    ax
        Optional Axes. If None, creates a new figure.
    figsize
        Figure size if creating a new figure.
    savepath
        Optional path to save the figure.
    show
        If True, call plt.show().

    Returns
    -------
    ax
        Axes with the bar plot drawn.
    """
    ax = _ensure_ax(ax, figsize=figsize)

    model_names = list(model_to_metric.keys())
    values = [model_to_metric[m] for m in model_names]

    indices = np.arange(len(model_names))
    ax.bar(indices, values)

    ax.set_xticks(indices)
    ax.set_xticklabels(model_names, rotation=20, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(f"Model Comparison ({metric_name})")
    ax.grid(True, axis="y", alpha=0.3)

    # show exact values on top of bars
    for idx, val in zip(indices, values):
        ax.text(idx, val, f"{val:.3f}", ha="center", va="bottom")

    fig = ax.get_figure()
    _maybe_save(fig, savepath)

    if show:
        plt.show()

    return ax

def plot_multi_precision_recall(
    y_true: np.ndarray,
    model_to_proba: Mapping[str, np.ndarray],
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot precision–recall curves for multiple models on a single Axes.

    Parameters
    ----------
    y_true
        Ground-truth binary labels (0/1) on the *same* test set.
    model_to_proba
        Mapping from model name to predicted probabilities on that test set.
    """
    y_true = np.asarray(y_true).ravel()
    ax = _ensure_ax(ax, figsize=figsize)

    for name, proba in model_to_proba.items():
        proba_arr = np.asarray(proba).ravel()
        precision, recall, _ = precision_recall_curve(y_true, proba_arr)
        ap = average_precision_score(y_true, proba_arr)
        ax.plot(recall, precision, linewidth=2, label=f"{name} (AP={ap:.3f})")

    ax.set_title("Precision–Recall Curves (Test)")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = ax.get_figure()
    _maybe_save(fig, savepath)
    if show:
        plt.show()
    return ax


def plot_multi_roc(
    y_true: np.ndarray,
    model_to_proba: Mapping[str, np.ndarray],
    *,
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot ROC curves for multiple models on a single Axes.
    """
    y_true = np.asarray(y_true).ravel()
    ax = _ensure_ax(ax, figsize=figsize)

    for name, proba in model_to_proba.items():
        proba_arr = np.asarray(proba).ravel()
        fpr, tpr, _ = roc_curve(y_true, proba_arr)
        roc_auc = roc_auc_score(y_true, proba_arr)
        ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC={roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1, alpha=0.5)

    ax.set_title("ROC Curves (Test)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = ax.get_figure()
    _maybe_save(fig, savepath)
    if show:
        plt.show()
    return ax


def plot_metric_over_epochs_multi(
    histories: Mapping[str, Mapping[str, Sequence[float]]],
    *,
    metric_key: str = "val_pr_auc",
    ax: Optional[plt.Axes] = None,
    figsize: Tuple[int, int] = (8, 6),
    savepath: Optional[Path | str] = None,
    show: bool = False,
) -> plt.Axes:
    """
    Plot a single metric (e.g. val_pr_auc) vs epoch for multiple models.

    Parameters
    ----------
    histories
        Mapping from model name to a history dict with keys:
          - "epoch" (optional; inferred if missing)
          - metric_key (e.g. "val_pr_auc")
    metric_key
        Key in each history dict to plot.
    """
    ax = _ensure_ax(ax, figsize=figsize)

    for name, hist in histories.items():
        epochs = list(hist.get("epoch", []))
        metric_vals = hist.get(metric_key)

        if metric_vals is None:
            continue

        if not epochs:
            n = len(metric_vals)
            epochs = list(range(1, n + 1))

        ax.plot(epochs, metric_vals, marker="o", label=name)

    ax.set_title(f"{metric_key} vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_key)
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig = ax.get_figure()
    _maybe_save(fig, savepath)
    if show:
        plt.show()
    return ax
