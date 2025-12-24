# src/ecom_fraud/evaluation/permutation_importance.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

try:
    import torch
except ImportError:  # pragma: no cover - torch is optional for non-torch use cases
    torch = None


__all__ = [
    "PermutationImportanceResult",
    "compute_permutation_importance_tabular",
    "compute_permutation_importance_torch_tabular",
    "pretty_print_importances",
]


# ---------------------------------------------------------------------------
# Dataclass to hold permutation importance results
# ---------------------------------------------------------------------------


@dataclass
class PermutationImportanceResult:
    """
    Container for permutation importance results on tabular data.

    Attributes
    ----------
    importances_df:
        DataFrame with one row per feature, columns:
            - feature: str, feature name
            - group: "numeric" or "categorical"
            - delta: baseline_score - mean(permuted_scores)
            - abs_delta: abs(delta)
    baseline_score:
        Metric value on the un-permuted data.
    metric_name:
        Name of the metric function used (for display / logging).
    """

    importances_df: pd.DataFrame
    baseline_score: float
    metric_name: str

    def top_k(self, k: int = 30) -> pd.DataFrame:
        """Return the top-k features by |Δ|."""
        return self.importances_df.sort_values("abs_delta", ascending=False).head(k)

    def bottom_k(self, k: int = 10) -> pd.DataFrame:
        """Return the bottom-k features by |Δ|."""
        return self.importances_df.sort_values("abs_delta", ascending=True).head(k)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_numpy(x: Any) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr


def _default_metric_name(metric: Callable[[np.ndarray, np.ndarray], float]) -> str:
    return getattr(metric, "__name__", "metric")


def _ensure_feature_names(
    n_num: int,
    n_cat: int,
    feature_names_num: Optional[Sequence[str]],
    feature_names_cat: Optional[Sequence[str]],
) -> Tuple[Sequence[str], Sequence[str]]:
    if feature_names_num is None:
        feature_names_num = [f"num_{j}" for j in range(n_num)]
    if feature_names_cat is None:
        feature_names_cat = [f"cat_{j}" for j in range(n_cat)]
    if len(feature_names_num) != n_num:
        raise ValueError(
            f"feature_names_num length {len(feature_names_num)} "
            f"does not match n_num={n_num}"
        )
    if len(feature_names_cat) != n_cat:
        raise ValueError(
            f"feature_names_cat length {len(feature_names_cat)} "
            f"does not match n_cat={n_cat}"
        )
    return feature_names_num, feature_names_cat


# ---------------------------------------------------------------------------
# Core: model-agnostic permutation importance on (X_num, X_cat)
# ---------------------------------------------------------------------------


def compute_permutation_importance_tabular(
    y_true: np.ndarray,
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    predict_proba_fn: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray],
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_repeats: int = 3,
    random_state: int = 42,
    feature_names_num: Optional[Sequence[str]] = None,
    feature_names_cat: Optional[Sequence[str]] = None,
) -> PermutationImportanceResult:
    """
    Compute permutation importance for tabular data with (numeric, categorical) blocks.

    This is fully model-agnostic: you just provide a `predict_proba_fn` that takes
    `(X_num, X_cat)` and returns predicted probabilities for the positive class.

    This is the core building block that you can use for:
      - Hybrid MLP (tree_score + embeddings)
      - Tree-only models (just pass X_cat=None and ignore cat features)
      - Any other tabular model where you can map arrays -> probabilities.

    Parameters
    ----------
    y_true
        Ground-truth labels, shape (n_samples,). Typically 0/1 for fraud.
    X_num
        Numeric feature matrix, shape (n_samples, n_num_features).
    X_cat
        Optional categorical feature matrix (e.g. embedding indices),
        shape (n_samples, n_cat_features). If None, only numeric features are used.
    predict_proba_fn
        Callable that maps `(X_num, X_cat)` -> 1D array of probabilities
        for the positive class. This is where you wrap your model inference.
    metric
        Metric function `metric(y_true, y_pred_proba)` used to compute the score.
        Defaults to `average_precision_score` (PR-AUC), which is appropriate
        for highly imbalanced fraud detection.
    n_repeats
        Number of random permutations per feature. The score drop is averaged
        over these repeats for robustness.
    random_state
        Seed for the NumPy RNG so results are reproducible.
    feature_names_num
        Optional list of names for numeric features. If None, generic names
        "num_0", "num_1", ... are used.
    feature_names_cat
        Optional list of names for categorical features. If None and X_cat is
        not None, generic names "cat_0", "cat_1", ... are used.

    Returns
    -------
    PermutationImportanceResult
        Dataclass with:
          - importances_df: one row per feature with Δ metric
          - baseline_score: metric on un-permuted data
          - metric_name: human-readable name of the metric
    """
    y_true = _to_numpy(y_true).ravel()
    X_num = _to_numpy(X_num)
    X_cat = _to_numpy(X_cat) if X_cat is not None else None

    n_samples, n_num_features = X_num.shape
    n_cat_features = 0 if X_cat is None else X_cat.shape[1]

    feature_names_num, feature_names_cat = _ensure_feature_names(
        n_num=n_num_features,
        n_cat=n_cat_features,
        feature_names_num=feature_names_num,
        feature_names_cat=feature_names_cat,
    )

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------
    # Baseline score
    # ------------------------------------------------------------------
    baseline_proba = predict_proba_fn(X_num, X_cat)
    baseline_proba = np.asarray(baseline_proba).reshape(-1)
    if baseline_proba.shape[0] != n_samples:
        raise ValueError(
            "predict_proba_fn must return an array of shape (n_samples,) "
            f"but got shape {baseline_proba.shape}"
        )

    baseline_score = float(metric(y_true, baseline_proba))
    metric_name = _default_metric_name(metric)

    # ------------------------------------------------------------------
    # Permutation importance
    # ------------------------------------------------------------------
    importances: Dict[str, float] = {}

    # Numeric features
    for j in range(n_num_features):
        scores: list[float] = []
        for _ in range(n_repeats):
            X_perm_num = X_num.copy()
            perm_idx = rng.permutation(n_samples)
            X_perm_num[:, j] = X_num[perm_idx, j]

            proba_perm = predict_proba_fn(X_perm_num, X_cat)
            proba_perm = np.asarray(proba_perm).reshape(-1)
            scores.append(float(metric(y_true, proba_perm)))

        importances[f"num::{feature_names_num[j]}"] = baseline_score - float(
            np.mean(scores)
        )

    # Categorical features (if any)
    if X_cat is not None and n_cat_features > 0:
        for j in range(n_cat_features):
            scores = []
            for _ in range(n_repeats):
                X_perm_cat = X_cat.copy()
                perm_idx = rng.permutation(n_samples)
                X_perm_cat[:, j] = X_cat[perm_idx, j]

                proba_perm = predict_proba_fn(X_num, X_perm_cat)
                proba_perm = np.asarray(proba_perm).reshape(-1)
                scores.append(float(metric(y_true, proba_perm)))

            importances[f"cat::{feature_names_cat[j]}"] = baseline_score - float(
                np.mean(scores)
            )

    # Build DataFrame
    rows = []
    for name, delta in importances.items():
        if name.startswith("num::"):
            group = "numeric"
            short = name[len("num::") :]
        elif name.startswith("cat::"):
            group = "categorical"
            short = name[len("cat::") :]
        else:
            group = "unknown"
            short = name

        rows.append(
            {
                "feature": short,
                "group": group,
                "delta": float(delta),
                "abs_delta": float(abs(delta)),
            }
        )

    importances_df = pd.DataFrame(rows).sort_values(
        "abs_delta", ascending=False
    ).reset_index(drop=True)

    return PermutationImportanceResult(
        importances_df=importances_df,
        baseline_score=baseline_score,
        metric_name=metric_name,
    )


# ---------------------------------------------------------------------------
# PyTorch-specific helper for your hybrid MLPs
# ---------------------------------------------------------------------------


def _torch_predict_proba_tabular(
    model: "torch.nn.Module",
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray],
    *,
    batch_size: int = 4096,
    device: Optional["torch.device"] = None,
) -> np.ndarray:
    """
    Internal helper: run a tabular PyTorch model on (X_num, X_cat) and return
    sigmoid probabilities for the positive class.

    Assumes your model signature is either:
      - model(x_num, x_cat) for embedding MLPs
      - model(x_num) for pure-numeric MLPs
    """
    if torch is None:
        raise ImportError(
            "torch is required for compute_permutation_importance_torch_tabular, "
            "but it is not installed."
        )

    model.eval()

    if device is None:
        device = next(model.parameters()).device

    X_num = np.asarray(X_num)
    X_cat = np.asarray(X_cat) if X_cat is not None else None

    n_samples = X_num.shape[0]
    probs_list: list[np.ndarray] = []

    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)

            x_num_batch = torch.from_numpy(X_num[start:end]).to(device=device)
            if X_cat is not None:
                x_cat_batch = torch.from_numpy(X_cat[start:end]).to(device=device)
                logits = model(x_num_batch, x_cat_batch)
            else:
                logits = model(x_num_batch)

            logits = logits.view(-1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)

    return np.concatenate(probs_list, axis=0)


def compute_permutation_importance_torch_tabular(
    model: "torch.nn.Module",
    y_true: np.ndarray,
    X_num: np.ndarray,
    X_cat: Optional[np.ndarray] = None,
    *,
    metric: Callable[[np.ndarray, np.ndarray], float] = average_precision_score,
    n_repeats: int = 3,
    random_state: int = 42,
    batch_size: int = 4096,
    device: Optional["torch.device"] = None,
    feature_names_num: Optional[Sequence[str]] = None,
    feature_names_cat: Optional[Sequence[str]] = None,
) -> PermutationImportanceResult:
    """
    Convenience wrapper for permutation importance with your PyTorch hybrid MLPs.

    This is designed specifically for your fraud project:
      - numeric features: include engineered features + tree_score
      - categorical features: embedding indices (country, bin_country, channel, etc.)
      - metric: PR-AUC by default (average_precision_score)

    Example usage (for your existing arrays):

        from ecom_fraud.evaluation.permutation_importance import (
            compute_permutation_importance_torch_tabular,
        )

        result_lgbm_hybrid = compute_permutation_importance_torch_tabular(
            model=best_model,                # your LGBM-hybrid embedding MLP
            y_true=y_test_np,
            X_num=X_test_num,
            X_cat=X_test_cat,
            metric=average_precision_score,
            n_repeats=3,
            batch_size=4096,
            device=device,
            feature_names_num=numeric_cols_for_mlp,
            feature_names_cat=embedding_cats,
        )

        result_lgbm_hybrid.top_k(30)
        result_lgbm_hybrid.bottom_k(10)

    Parameters
    ----------
    model
        Trained PyTorch model with signature `model(x_num, x_cat)` or `model(x_num)`.
    y_true
        Ground-truth labels (0/1), shape (n_samples,).
    X_num
        Numeric feature matrix used by the model, shape (n_samples, n_num_features).
    X_cat
        Optional categorical feature matrix (e.g. embedding indices),
        shape (n_samples, n_cat_features).
    metric
        Metric function `metric(y_true, y_pred_proba)` (defaults to PR-AUC).
    n_repeats
        Number of permutations per feature.
    random_state
        RNG seed for reproducible permutations.
    batch_size
        Batch size for inference on the model.
    device
        torch.device on which the model should run. If None, uses model's device.
    feature_names_num
        Optional list of numeric feature names. Typically your `numeric_cols_for_mlp`
        (including "tree_score").
    feature_names_cat
        Optional list of categorical feature names. Typically your `embedding_cats`.

    Returns
    -------
    PermutationImportanceResult
        See `PermutationImportanceResult` docstring.
    """

    if torch is None:
        raise ImportError(
            "torch is required for compute_permutation_importance_torch_tabular, "
            "but it is not installed."
        )

    def predict_proba_fn(Xn: np.ndarray, Xc: Optional[np.ndarray]) -> np.ndarray:
        return _torch_predict_proba_tabular(
            model=model,
            X_num=Xn,
            X_cat=Xc,
            batch_size=batch_size,
            device=device,
        )

    return compute_permutation_importance_tabular(
        y_true=y_true,
        X_num=X_num,
        X_cat=X_cat,
        predict_proba_fn=predict_proba_fn,
        metric=metric,
        n_repeats=n_repeats,
        random_state=random_state,
        feature_names_num=feature_names_num,
        feature_names_cat=feature_names_cat,
    )


# ---------------------------------------------------------------------------
# Pretty-print helper for notebooks / reports
# ---------------------------------------------------------------------------


def pretty_print_importances(
    result: PermutationImportanceResult,
    top_k: int = 30,
    bottom_k: int = 10,
) -> None:
    """
    Console-style pretty print for top and bottom features.

    This mirrors what you were doing in the notebook, but centralized here so
    you can just call it for both LGBM-hybrid and XGB-hybrid.

    Parameters
    ----------
    result
        PermutationImportanceResult from one of the compute_* functions.
    top_k
        Number of top features (by |Δ|) to print.
    bottom_k
        Number of least important features to print.
    """
    df = result.importances_df

    print(
        f"\nBaseline {result.metric_name}: {result.baseline_score:.6f} "
        "(higher is better)\n"
    )

    # Top k by absolute delta
    top = df.sort_values("abs_delta", ascending=False).head(top_k)
    print(f"Top {len(top)} features by permutation importance (Δ{result.metric_name}):")
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        print(
            f"{rank:2d}. {row.group:10s} {row.feature:30s} "
            f"Δ={row.delta:+.5f} |Δ|={row.abs_delta:.5f}"
        )

    # Bottom k (candidates for pruning)
    bottom = df.sort_values("abs_delta", ascending=True).head(bottom_k)
    print(f"\nLeast important {len(bottom)} features (candidates for pruning):")
    for row in bottom.itertuples(index=False):
        print(
            f"-  {row.group:10s} {row.feature:30s} "
            f"Δ={row.delta:+.5f} |Δ|={row.abs_delta:.5f}"
        )
