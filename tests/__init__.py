"""
Test suite for the ml_tabular package.

This package is organized by concern:

- data/       – tests for datasets, loading, and preprocessing
- models/     – tests for model architectures and configs
- training/   – tests for training loops, early stopping, etc.
- cli/        – tests for the Typer-based command-line interface
- mlops/      – tests for MLflow and other MLOps utilities
- conftest.py – shared fixtures and test configuration

Notes
-----
- This module is intentionally minimal:
  * No side effects (no imports that configure logging, load configs, etc.).
  * No automatic test discovery tweaks.
- Pytest discovers tests based on file names (test_*.py), not by importing
  the tests package directly, so this file is primarily documentation.
"""

# Expose no public API from the tests package. Tests are meant to be
# discovered and run by pytest, not imported as a library.
__all__: list[str] = []
