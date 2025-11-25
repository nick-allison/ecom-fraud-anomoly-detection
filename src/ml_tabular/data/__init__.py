"""
Data access utilities for ml_tabular.

Typical contents of this package include:

- Helpers for reading local files into pandas DataFrames (CSV/Parquet/etc.).
- Adapters for external sources (e.g. Kaggle downloads, SQL databases, document stores).
- Lightweight wrappers that keep IO concerns separate from feature engineering
  and modeling code.

Import the concrete modules directly, for example:

    from ml_tabular.data.kaggle import download_kaggle_dataset

to keep dependencies and side effects explicit.
"""

from __future__ import annotations

__all__: list[str] = []
