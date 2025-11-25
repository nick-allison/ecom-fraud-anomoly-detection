from __future__ import annotations

from typing import Iterable

import pytest
import torch
from torch.utils.data import DataLoader

from ml_tabular.exceptions import DataError
from ml_tabular.torch.datasets.tabular import TabularDataset


# ---------------------------------------------------------------------------
# Basic construction from a DataFrame
# ---------------------------------------------------------------------------


def test_tabular_dataset_from_dataframe_basic(simple_tabular_df) -> None:
    """TabularDataset.from_dataframe should build a usable dataset from a small DataFrame."""
    df = simple_tabular_df

    ds = TabularDataset.from_dataframe(
        df,
        feature_columns=["feat1", "feat2"],
        target_column="target",
    )

    # Dataset length matches number of rows
    assert len(ds) == len(df)

    # A single item gives (x, y) tensors with expected shapes and dtypes
    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # Features: 1D tensor with len == number of features
    assert x.dim() == 1
    assert x.shape[0] == 2

    # Targets: allow scalar or 1D of length 1 (depending on implementation)
    assert y.dim() in (0, 1)
    if y.dim() == 1:
        assert y.shape[0] == 1

    # Dtypes: typically float32 for tabular features/targets
    assert x.dtype == torch.float32
    assert y.dtype in (torch.float32, torch.long, torch.int64)


def test_tabular_dataset_metadata_is_populated(simple_tabular_df) -> None:
    """Metadata should describe feature names, target name, and dimensions."""
    df = simple_tabular_df

    ds = TabularDataset.from_dataframe(
        df,
        feature_columns=["feat1", "feat2"],
        target_column="target",
    )

    meta = ds.metadata

    # Basic presence checks
    assert meta.num_samples == len(ds)
    assert meta.num_features == 2
    assert meta.feature_names == ["feat1", "feat2"]
    assert meta.target_name == "target"


# ---------------------------------------------------------------------------
# Feature inference and robustness
# ---------------------------------------------------------------------------


def test_tabular_dataset_can_infer_feature_columns(simple_tabular_df) -> None:
    """If feature_columns is omitted, the dataset should be able to infer them."""
    df = simple_tabular_df

    # Assuming from_dataframe can infer feature columns when not provided
    ds = TabularDataset.from_dataframe(
        df,
        feature_columns=None,  # type: ignore[arg-type]
        target_column="target",
    )

    x, y = ds[0]

    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    # We still expect 2 features (all columns except target)
    assert x.shape[-1] == 2

    meta = ds.metadata
    # It's reasonable to expect the same inferred feature column names
    assert meta.feature_names == ["feat1", "feat2"]
    assert meta.target_name == "target"


def test_tabular_dataset_missing_feature_column_raises(simple_tabular_df) -> None:
    """If a requested feature column is missing, a DataError should be raised."""
    df = simple_tabular_df

    with pytest.raises(DataError) as ctx:
        TabularDataset.from_dataframe(
            df,
            feature_columns=["feat1", "does_not_exist"],
            target_column="target",
        )

    exc = ctx.value
    # Don't be too brittle: just ensure it's a DataError and the message mentions the bad column
    assert isinstance(exc, DataError)
    text = str(exc)
    assert "does_not_exist" in text or "missing" in text.lower()


def test_tabular_dataset_missing_target_column_raises(simple_tabular_df) -> None:
    """If the target column is missing, a DataError should be raised."""
    df = simple_tabular_df.rename(columns={"target": "label"})

    with pytest.raises(DataError):
        TabularDataset.from_dataframe(
            df,
            feature_columns=["feat1", "feat2"],
            target_column="target",  # no longer present
        )


# ---------------------------------------------------------------------------
# DataLoader integration
# ---------------------------------------------------------------------------


def test_tabular_dataset_works_with_dataloader(simple_tabular_df) -> None:
    """TabularDataset should work seamlessly with torch.utils.data.DataLoader."""
    df = simple_tabular_df

    ds = TabularDataset.from_dataframe(
        df,
        feature_columns=["feat1", "feat2"],
        target_column="target",
    )

    batch_size = 2
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    batches = list(iter(loader))

    # Number of batches: ceil(len(ds) / batch_size)
    assert len(batches) == ((len(ds) + batch_size - 1) // batch_size)

    x_batch, y_batch = batches[0]

    assert isinstance(x_batch, torch.Tensor)
    assert isinstance(y_batch, torch.Tensor)

    # Batch shapes
    assert x_batch.dim() == 2
    assert y_batch.dim() in (1, 2)

    assert x_batch.shape[0] <= batch_size
    assert x_batch.shape[1] == 2  # num_features

    # If y is 2D, treat last dim as 1
    if y_batch.dim() == 2:
        assert y_batch.shape[1] in (1,)

    # Dtypes
    assert x_batch.dtype == torch.float32
    assert y_batch.dtype in (torch.float32, torch.long, torch.int64)


def test_tabular_dataset_is_indexable_like_a_sequence(simple_tabular_df) -> None:
    """Dataset should behave like a sequence: supports __len__ and __getitem__ for indices."""
    df = simple_tabular_df
    ds = TabularDataset.from_dataframe(
        df,
        feature_columns=["feat1", "feat2"],
        target_column="target",
    )

    # Index all items and ensure we get the same number back
    items = [ds[i] for i in range(len(ds))]
    assert len(items) == len(df)

    for x, y in items:
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
