"""
Dataset definitions for PyTorch-based training.

Contains modules like:

- tabular: TabularDataset for dense numeric features.
- time_series: TimeSeriesSequenceDataset for sliding-window forecasting.

These datasets sit on top of preprocessed numpy/pandas data and expose
PyTorch-friendly shapes for use with DataLoader.
"""

from __future__ import annotations

__all__: list[str] = []
