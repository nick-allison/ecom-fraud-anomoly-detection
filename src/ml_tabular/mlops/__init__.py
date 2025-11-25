"""
MLOps and experiment-tracking helpers for ml_tabular.

This package is intended for:

- MLflow integration (starting runs, logging parameters and metrics).
- Future additions like model registry helpers or deployment utilities.

Keeping these concerns separate from core modeling code makes it easy to
toggle MLOps tooling on or off in different environments.
"""

from __future__ import annotations

__all__: list[str] = []
