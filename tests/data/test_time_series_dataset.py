from __future__ import annotations

import math

import pytest
import torch
from torch.utils.data import DataLoader

from ml_tabular.exceptions import DataError
from ml_tabular.torch.datasets.time_series import TimeSeriesSequenceDataset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _expected_num_windows(
    n: int,
    input_window: int,
    prediction_horizon: int,
    step_size: int,
) -> int:
    """Compute the expected number of sliding windows for tests.

    We assume the dataset logic is:

      - Use contiguous windows of length `input_window` for inputs.
      - Follow them by a prediction horizon of `prediction_horizon` steps.
      - Start indices are 0, step_size, 2 * step_size, ...
      - A window is valid if `start + input_window + prediction_horizon <= n`.

    That yields:

      max_start = n - (input_window + prediction_horizon)
      if max_start < 0: 0 windows
      else: floor(max_start / step_size) + 1
    """
    max_start = n - (input_window + prediction_horizon)
    if max_start < 0:
        return 0
    return max_start // step_size + 1


# ---------------------------------------------------------------------------
# Basic construction and shapes
# ---------------------------------------------------------------------------


def test_time_series_dataset_basic_shapes(simple_time_series_df) -> None:
    """from_dataframe should build a sequence dataset with correct shapes."""
    df = simple_time_series_df

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=3,
        prediction_horizon=2,
        step_size=1,
    )

    assert len(ds) > 0

    x, y = ds[0]

    # x: (input_window, num_features)
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, 2)

    # y: (prediction_horizon, num_targets) or (prediction_horizon,) depending on your design
    assert isinstance(y, torch.Tensor)
    assert y.dim() in (1, 2)
    if y.dim() == 2:
        assert y.shape[0] == 2
        assert y.shape[1] == 1
    else:
        assert y.shape[0] == 2

    # dtypes
    assert x.dtype == torch.float32
    assert y.dtype in (torch.float32, torch.long, torch.int64)


def test_time_series_dataset_metadata(simple_time_series_df) -> None:
    """Metadata should describe the windowing setup and feature/target dimensions."""
    df = simple_time_series_df

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=4,
        prediction_horizon=1,
        step_size=2,
    )

    meta = ds.metadata

    assert meta.input_window == 4
    assert meta.prediction_horizon == 1
    assert meta.step_size == 2
    assert meta.num_features == 2
    assert meta.num_targets == 1
    assert meta.num_samples == len(ds)

    # These are optional but nice if you expose them
    if hasattr(meta, "value_columns"):
        assert list(meta.value_columns) == ["feat1", "feat2"]
    if hasattr(meta, "target_column"):
        assert meta.target_column == "target"
    if hasattr(meta, "datetime_column"):
        assert meta.datetime_column == "timestamp"


# ---------------------------------------------------------------------------
# Window count and step_size behaviour
# ---------------------------------------------------------------------------


def test_time_series_dataset_window_count_with_step_size(simple_time_series_df) -> None:
    """Number of windows should match the expected sliding-window formula."""
    df = simple_time_series_df
    n = len(df)
    input_window = 3
    horizon = 2
    step_size = 1

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=input_window,
        prediction_horizon=horizon,
        step_size=step_size,
    )

    expected = _expected_num_windows(n, input_window, horizon, step_size)
    assert len(ds) == expected


def test_time_series_dataset_window_count_with_larger_step(simple_time_series_df) -> None:
    """Larger step_size should reduce the number of windows."""
    df = simple_time_series_df
    n = len(df)
    input_window = 3
    horizon = 1
    step_size = 2

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=input_window,
        prediction_horizon=horizon,
        step_size=step_size,
    )

    expected = _expected_num_windows(n, input_window, horizon, step_size)
    assert len(ds) == expected


def test_time_series_dataset_handles_too_large_windows(simple_time_series_df) -> None:
    """If the window + horizon exceed the sequence length, dataset may be empty, but should not crash."""
    df = simple_time_series_df
    n = len(df)

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=n,
        prediction_horizon=5,
        step_size=1,
    )

    # With input_window >= n and horizon > 0, it's reasonable to have 0 windows
    assert len(ds) == 0


# ---------------------------------------------------------------------------
# Sequence semantics: x and y should align with the underlying data
# ---------------------------------------------------------------------------


def test_time_series_dataset_sequence_alignment(simple_time_series_df) -> None:
    """For our synthetic fixture, we can sanity-check the actual numeric values."""
    df = simple_time_series_df

    # simple_time_series_df has:
    # feat1 = 0, 1, 2, ...
    # feat2 = 0, 2, 4, ...
    # target = i % 3
    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=3,
        prediction_horizon=2,
        step_size=1,
    )

    x, y = ds[0]

    # x should contain rows for indices 0,1,2 for feat1/feat2
    # So first feature column should be [0,1,2]
    assert torch.allclose(x[:, 0], torch.tensor([0.0, 1.0, 2.0]))
    # Second feature column should be [0,2,4]
    assert torch.allclose(x[:, 1], torch.tensor([0.0, 2.0, 4.0]))

    # y should correspond to target at indices 3 and 4 => 3 % 3 = 0, 4 % 3 = 1
    y_values = y.flatten().to(torch.float32)
    assert torch.allclose(y_values, torch.tensor([0.0, 1.0]))


# ---------------------------------------------------------------------------
# Column validation and errors
# ---------------------------------------------------------------------------


def test_time_series_dataset_missing_value_column_raises(simple_time_series_df) -> None:
    """Missing value columns should raise DataError."""
    df = simple_time_series_df

    with pytest.raises(DataError) as ctx:
        TimeSeriesSequenceDataset.from_dataframe(
            df,
            value_columns=["feat1", "does_not_exist"],
            target_column="target",
            sort_by="timestamp",
            dropna=True,
            input_window=3,
            prediction_horizon=1,
            step_size=1,
        )

    exc = ctx.value
    assert isinstance(exc, DataError)
    text = str(exc)
    assert "does_not_exist" in text or "missing" in text.lower()


def test_time_series_dataset_missing_target_column_raises(simple_time_series_df) -> None:
    """Missing target column should raise DataError."""
    df = simple_time_series_df.rename(columns={"target": "label"})

    with pytest.raises(DataError):
        TimeSeriesSequenceDataset.from_dataframe(
            df,
            value_columns=["feat1", "feat2"],
            target_column="target",  # doesn't exist anymore
            sort_by="timestamp",
            dropna=True,
            input_window=3,
            prediction_horizon=1,
            step_size=1,
        )


def test_time_series_dataset_missing_datetime_column_raises(simple_time_series_df) -> None:
    """If sort_by refers to a non-existent datetime column, DataError should be raised."""
    df = simple_time_series_df

    with pytest.raises(DataError):
        TimeSeriesSequenceDataset.from_dataframe(
            df,
            value_columns=["feat1", "feat2"],
            target_column="target",
            sort_by="does_not_exist",
            dropna=True,
            input_window=3,
            prediction_horizon=1,
            step_size=1,
        )


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


def test_time_series_dataset_works_with_dataloader(simple_time_series_df) -> None:
    """TimeSeriesSequenceDataset should integrate cleanly with DataLoader."""
    df = simple_time_series_df

    ds = TimeSeriesSequenceDataset.from_dataframe(
        df,
        value_columns=["feat1", "feat2"],
        target_column="target",
        sort_by="timestamp",
        dropna=True,
        input_window=3,
        prediction_horizon=1,
        step_size=1,
    )

    if len(ds) == 0:
        pytest.skip("Dataset has zero windows for this configuration; cannot test DataLoader.")

    batch_size = 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    batches = list(iter(loader))
    assert len(batches) == math.ceil(len(ds) / batch_size)

    x_batch, y_batch = batches[0]

    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)

    # x_batch: (batch_size, input_window, num_features)
    assert x_batch.dim() == 3
    assert x_batch.shape[1] == 3  # input_window
    assert x_batch.shape[2] == 2  # num_features

    # y_batch: (batch_size, horizon, num_targets) or (batch_size, horizon)
    assert y_batch.dim() in (2, 3)
    if y_batch.dim() == 3:
        assert y_batch.shape[1] == 1 or y_batch.shape[1] == 1  # horizon
    # Dtypes
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype in (torch.float32, torch.long, torch.int64)
