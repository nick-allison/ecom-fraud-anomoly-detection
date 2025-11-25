"""
Feature engineering utilities for ml_tabular.

This package typically includes:

- Reusable feature transformations for tabular data
  (e.g., build_basic_features, date decompositions).
- Optional domain-specific feature builders for particular datasets or problems.

The goal is to keep all feature logic in one place so that training scripts
remain small and declarative.
"""

from __future__ import annotations

__all__: list[str] = []
