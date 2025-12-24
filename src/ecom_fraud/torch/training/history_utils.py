# ecom_fraud/torch/training/history_utils.py

from __future__ import annotations
from typing import Any, Dict, List, Optional


def get_best_history(search_result):
    """
    Extract the per-epoch history dict from a RandomSearchResult
    returned by `run_tabular_embedding_random_search`.

    We normalize to a dict-of-lists so it works with plot_training_curves.
    Returns None if no history is attached.
    """

    # 1) Preferred: RandomSearchResult.best_history (we just added this)
    hist = getattr(search_result, "best_history", None)

    # 2) Fallback: go through best_trial.fit_result["history"]
    if hist is None:
        best_trial = getattr(search_result, "best_trial", None)
        if best_trial is not None:
            hist = getattr(best_trial, "history", None)

    if hist is None:
        print(
            f"[get_best_history] No training history found on "
            f"{type(search_result).__name__}; skipping learning-curve plots."
        )
        return None

    # --- Normalize: dict-of-lists is fine as-is --------------------
    if isinstance(hist, dict):
        return hist

    # If it's a list of per-epoch dicts, convert to dict-of-lists
    if isinstance(hist, (list, tuple)) and len(hist) > 0 and isinstance(hist[0], dict):
        keys = hist[0].keys()
        normalized = {k: [row[k] for row in hist] for k in keys}
        return normalized

    print(
        f"[get_best_history] Unsupported history type {type(hist)}; "
        "expected dict or list-of-dicts. Skipping."
    )
    return None
