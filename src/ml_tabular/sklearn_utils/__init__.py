"""
Scikit-learn utilities for ml_tabular.

This package usually contains:

- Pipeline builders (e.g., functions that construct ColumnTransformer pipelines
  for numerical + categorical features).
- Encoders, imputers, and standard preprocessing steps wrapped in a consistent,
  testable way.

Import the concrete helper modules directly, e.g.:

    from ml_tabular.sklearn_utils.tabular_pipeline import make_tabular_pipeline
"""

from __future__ import annotations

__all__: list[str] = []
