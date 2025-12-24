from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecom_fraud.exceptions import DataError
from ecom_fraud.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularDatasetMetadata:
    """Lightweight metadata about a tabular dataset.

    Attributes
    ----------
    num_features:
        Number of input features per sample.
    feature_names:
        Optional list of feature names, typically derived from a DataFrame's columns.
    has_targets:
        Whether this dataset includes target values (y). For pure inference datasets,
        this will be False.
    """

    num_features: int
    feature_names: Optional[List[str]] = None
    has_targets: bool = True


@dataclass(frozen=True)
class EmbeddingTabularDatasetMetadata:
    """Metadata for an embedding-style tabular dataset with numeric + categorical.

    Attributes
    ----------
    num_numeric_features:
        Number of numeric input features per sample.
    num_categorical_features:
        Number of categorical (index) input features per sample.
    numeric_feature_names:
        Optional feature names for numeric features.
    categorical_feature_names:
        Optional feature names for categorical features.
    has_targets:
        Whether this dataset includes target values (y).
    """

    num_numeric_features: int
    num_categorical_features: int
    numeric_feature_names: Optional[List[str]] = None
    categorical_feature_names: Optional[List[str]] = None
    has_targets: bool = True


# ---------------------------------------------------------------------------
# Core dense Dataset
# ---------------------------------------------------------------------------


class TabularDataset(Dataset):
    """PyTorch Dataset for tabular data.

    This Dataset is designed to sit at the boundary between your data/feature
    layer and your model training code. It assumes that feature engineering
    (e.g., scaling, one-hot encoding) has already been applied, and expects
    a dense numeric feature matrix (numpy array or DataFrame).

    Typical usage
    -------------
    From numpy arrays:

        X_train: np.ndarray  # shape (N, D)
        y_train: np.ndarray  # shape (N,) or (N, 1)

        ds_train = TabularDataset.from_arrays(X_train, y_train)

    From a pandas DataFrame:

        df = ...
        ds_train = TabularDataset.from_dataframe(
            df,
            target_column="target",
            drop_columns=["id"],
        )

    In your training script:

        train_loader = DataLoader(ds_train, batch_size=64, shuffle=True)

        for batch in train_loader:
            if ds_train.metadata.has_targets:
                X_batch, y_batch = batch
            else:
                X_batch = batch
            ...

    Notes
    -----
    - Features are stored as float32 tensors by default.
    - Targets default to float32 for regression-like tasks if y is float-like,
      or long (int64) if y looks categorical/integer. You can override this
      explicitly via target_dtype.
    """

    def __init__(
        self,
        features: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        *,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        X_np, inferred_feature_names = self._to_2d_numpy(features)
        y_np = self._targets_to_numpy(targets) if targets is not None else None

        if y_np is not None and X_np.shape[0] != y_np.shape[0]:
            raise DataError(
                "Features and targets must have the same number of rows.",
                code="tabular_shapes_mismatch",
                context={
                    "n_samples_features": int(X_np.shape[0]),
                    "n_samples_targets": int(y_np.shape[0]),
                },
                location="ml_tabular.torch.datasets.tabular.TabularDataset.__init__",
            )

        if X_np.ndim != 2:
            raise DataError(
                "Features must be a 2D array of shape (n_samples, n_features).",
                code="tabular_features_not_2d",
                context={"shape": X_np.shape},
                location="ml_tabular.torch.datasets.tabular.TabularDataset.__init__",
            )

        # Choose feature names: user-provided wins, otherwise inferred from DataFrame
        final_feature_names: Optional[List[str]]
        if feature_names is not None:
            final_feature_names = list(feature_names)
        else:
            final_feature_names = inferred_feature_names

        if final_feature_names is not None and len(final_feature_names) != X_np.shape[1]:
            raise DataError(
                "Length of feature_names does not match number of feature columns.",
                code="tabular_feature_names_mismatch",
                context={
                    "num_features": int(X_np.shape[1]),
                    "len_feature_names": len(final_feature_names),
                },
                location="ml_tabular.torch.datasets.tabular.TabularDataset.__init__",
            )

        # Convert to tensors
        self._X = torch.as_tensor(X_np, dtype=feature_dtype)

        if y_np is not None:
            if target_dtype is None:
                target_dtype = self._infer_target_dtype(y_np)
            self._y = torch.as_tensor(y_np, dtype=target_dtype)
        else:
            self._y = None

        self._metadata = TabularDatasetMetadata(
            num_features=int(self._X.shape[1]),
            feature_names=final_feature_names,
            has_targets=self._y is not None,
        )

        logger.info(
            "Created TabularDataset: n_samples=%d, n_features=%d, has_targets=%s",
            self._X.shape[0],
            self._X.shape[1],
            self._metadata.has_targets,
        )

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_2d_numpy(
        features: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Convert features to a 2D numpy array and optionally extract feature names."""
        if isinstance(features, pd.DataFrame):
            if features.empty:
                raise DataError(
                    "Feature DataFrame is empty.",
                    code="tabular_features_empty",
                    context={},
                    location="ml_tabular.torch.datasets.tabular.TabularDataset._to_2d_numpy",
                )
            feature_names = list(features.columns)
            X_np = features.to_numpy()
        elif isinstance(features, np.ndarray):
            if features.size == 0:
                raise DataError(
                    "Feature array is empty.",
                    code="tabular_features_empty",
                    context={},
                    location="ml_tabular.torch.datasets.tabular.TabularDataset._to_2d_numpy",
                )
            if features.ndim == 1:
                # Interpret as a single feature; reshape to (N, 1)
                X_np = features.reshape(-1, 1)
            else:
                X_np = features
            feature_names = None
        else:
            raise DataError(
                "Unsupported type for features; expected numpy.ndarray or pandas.DataFrame.",
                code="tabular_features_bad_type",
                context={"type": type(features).__name__},
                location="ml_tabular.torch.datasets.tabular.TabularDataset._to_2d_numpy",
            )

        return np.asarray(X_np), feature_names

    @staticmethod
    def _targets_to_numpy(
        targets: Union[np.ndarray, pd.Series, pd.DataFrame],
    ) -> np.ndarray:
        """Convert targets to a 1D numpy array."""
        if isinstance(targets, pd.DataFrame):
            if targets.shape[1] != 1:
                raise DataError(
                    "Target DataFrame must have exactly one column.",
                    code="tabular_targets_multi_column",
                    context={"shape": targets.shape},
                    location="ml_tabular.torch.datasets.tabular.TabularDataset._targets_to_numpy",
                )
            y_np = targets.iloc[:, 0].to_numpy()
        elif isinstance(targets, pd.Series):
            y_np = targets.to_numpy()
        elif isinstance(targets, np.ndarray):
            if targets.ndim == 2 and targets.shape[1] == 1:
                y_np = targets.reshape(-1)
            elif targets.ndim == 1:
                y_np = targets
            else:
                raise DataError(
                    "Target array must be 1D or 2D with a single column.",
                    code="tabular_targets_bad_shape",
                    context={"shape": targets.shape},
                    location="ml_tabular.torch.datasets.tabular.TabularDataset._targets_to_numpy",
                )
        else:
            raise DataError(
                "Unsupported type for targets; expected numpy.ndarray, pandas.Series, or pandas.DataFrame.",
                code="tabular_targets_bad_type",
                context={"type": type(targets).__name__},
                location="ml_tabular.torch.datasets.tabular.TabularDataset._targets_to_numpy",
            )

        if y_np.size == 0:
            raise DataError(
                "Target array is empty.",
                code="tabular_targets_empty",
                context={},
                location="ml_tabular.torch.datasets.tabular.TabularDataset._targets_to_numpy",
            )

        return np.asarray(y_np)

    @staticmethod
    def _infer_target_dtype(y_np: np.ndarray) -> torch.dtype:
        """Infer a reasonable torch dtype for targets based on their numpy dtype."""
        if np.issubdtype(y_np.dtype, np.integer):
            return torch.long
        if np.issubdtype(y_np.dtype, np.floating):
            return torch.float32
        # Fallback: treat as float32 but warn
        logger.warning(
            "Target dtype %s not clearly integer or float; defaulting to float32.",
            y_np.dtype,
        )
        return torch.float32

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._X.shape[0]

    def __getitem__(self, idx: int):
        if self._y is None:
            return self._X[idx]
        return self._X[idx], self._y[idx]

    # ------------------------------------------------------------------
    # Public properties & alternative constructors
    # ------------------------------------------------------------------

    @property
    def features(self) -> torch.Tensor:
        """Return the feature tensor X (shape: [n_samples, n_features])."""
        return self._X

    @property
    def targets(self) -> Optional[torch.Tensor]:
        """Return the target tensor y (shape: [n_samples]) or None if not present."""
        return self._y

    @property
    def metadata(self) -> TabularDatasetMetadata:
        """Return metadata about this dataset (num_features, feature_names, etc.)."""
        return self._metadata

    @classmethod
    def from_arrays(
        cls,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
        feature_names: Optional[Sequence[str]] = None,
    ) -> "TabularDataset":
        """Construct a TabularDataset directly from numpy arrays.

        Parameters
        ----------
        X:
            Feature matrix of shape (n_samples, n_features) or (n_samples,).
        y:
            Optional target vector of shape (n_samples,) or (n_samples, 1).
        feature_dtype:
            torch.dtype for features (default: float32).
        target_dtype:
            Optional torch.dtype for targets; if None, it will be inferred.
        feature_names:
            Optional explicit feature names.

        Returns
        -------
        TabularDataset
        """
        return cls(
            features=X,
            targets=y,
            feature_dtype=feature_dtype,
            target_dtype=target_dtype,
            feature_names=feature_names,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        target_column: Optional[str] = None,
        drop_columns: Sequence[str] = (),
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
    ) -> "TabularDataset":
        """Construct a TabularDataset from a pandas DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame containing features (and optionally a target column).
        target_column:
            Name of the target column. If None, the dataset will be created
            without targets (has_targets=False).
        drop_columns:
            Columns to drop from the feature matrix (e.g., IDs).
        feature_dtype:
            torch.dtype for features (default: float32).
        target_dtype:
            Optional torch.dtype for targets; if None, it will be inferred.

        Returns
        -------
        TabularDataset
            Dataset with X derived from feature columns and y from target_column
            (if provided).

        Raises
        ------
        DataError
            If target_column is specified but not present, or if no feature
            columns remain after dropping.
        """
        if df.empty:
            raise DataError(
                "Input DataFrame is empty.",
                code="tabular_df_empty",
                context={},
                location="ml_tabular.torch.datasets.tabular.TabularDataset.from_dataframe",
            )

        df = df.copy()

        y = None
        if target_column is not None:
            if target_column not in df.columns:
                raise DataError(
                    f"Target column '{target_column}' not found in DataFrame.",
                    code="tabular_target_missing",
                    context={"columns": list(df.columns)},
                    location="ml_tabular.torch.datasets.tabular.TabularDataset.from_dataframe",
                )
            y = df[target_column]
            df = df.drop(columns=[target_column])

        # Drop unwanted columns from features
        to_drop = [c for c in drop_columns if c in df.columns]
        if to_drop:
            df = df.drop(columns=to_drop)

        if df.shape[1] == 0:
            raise DataError(
                "No feature columns remain after dropping target and drop_columns.",
                code="tabular_no_features_after_drop",
                context={
                    "target_column": target_column,
                    "drop_columns": list(drop_columns),
                },
                location="ml_tabular.torch.datasets.tabular.TabularDataset.from_dataframe",
            )

        feature_names = list(df.columns)

        return cls(
            features=df,
            targets=y,
            feature_dtype=feature_dtype,
            target_dtype=target_dtype,
            feature_names=feature_names,
        )


# ---------------------------------------------------------------------------
# Embedding-style tabular Dataset (numeric + categorical)
# ---------------------------------------------------------------------------


class EmbeddingTabularDataset(Dataset):
    """Dataset for numeric + categorical tabular features.

    This is intended for embedding-style models where you have:

        - X_num: dense numeric features (already scaled / engineered)
        - X_cat: categorical features represented as integer indices
        - y:     optional targets

    The key behavior is the __getitem__ signature:

        - If targets are present:
              ((x_num, x_cat), y)
        - If targets are absent:
              (x_num, x_cat)

    This structure is designed to work cleanly with the generic
    `_move_batch_to_device` logic in the training loops, which can
    recursively move tuples/lists/dicts of tensors to a device.
    """

    def __init__(
        self,
        X_num: Union[np.ndarray, pd.DataFrame],
        X_cat: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        *,
        numeric_feature_dtype: torch.dtype = torch.float32,
        categorical_feature_dtype: torch.dtype = torch.long,
        target_dtype: Optional[torch.dtype] = None,
        numeric_feature_names: Optional[Sequence[str]] = None,
        categorical_feature_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        # Reuse the conversion logic from TabularDataset for consistency
        X_num_np, inferred_num_names = TabularDataset._to_2d_numpy(X_num)
        X_cat_np, inferred_cat_names = TabularDataset._to_2d_numpy(X_cat)
        y_np = (
            TabularDataset._targets_to_numpy(targets) if targets is not None else None
        )

        # Check row alignment
        n_samples_num = X_num_np.shape[0]
        n_samples_cat = X_cat_np.shape[0]

        if n_samples_num != n_samples_cat:
            raise DataError(
                "Numeric and categorical feature blocks must have the same number of rows.",
                code="embedding_tabular_shapes_mismatch_features",
                context={
                    "n_samples_numeric": int(n_samples_num),
                    "n_samples_categorical": int(n_samples_cat),
                },
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        if y_np is not None and n_samples_num != y_np.shape[0]:
            raise DataError(
                "Features and targets must have the same number of rows.",
                code="embedding_tabular_shapes_mismatch_targets",
                context={
                    "n_samples_features": int(n_samples_num),
                    "n_samples_targets": int(y_np.shape[0]),
                },
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        # Enforce 2D for both numeric and categorical features
        if X_num_np.ndim != 2:
            raise DataError(
                "Numeric features must be a 2D array of shape (n_samples, n_num_features).",
                code="embedding_tabular_num_not_2d",
                context={"shape": X_num_np.shape},
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        if X_cat_np.ndim != 2:
            raise DataError(
                "Categorical features must be a 2D array of shape (n_samples, n_cat_features).",
                code="embedding_tabular_cat_not_2d",
                context={"shape": X_cat_np.shape},
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        # Resolve feature names (user-specified overrides inferred)
        if numeric_feature_names is not None:
            final_num_names = list(numeric_feature_names)
        else:
            final_num_names = inferred_num_names

        if categorical_feature_names is not None:
            final_cat_names = list(categorical_feature_names)
        else:
            final_cat_names = inferred_cat_names

        if final_num_names is not None and len(final_num_names) != X_num_np.shape[1]:
            raise DataError(
                "Length of numeric_feature_names does not match number of numeric feature columns.",
                code="embedding_tabular_num_feature_names_mismatch",
                context={
                    "num_numeric_features": int(X_num_np.shape[1]),
                    "len_numeric_feature_names": len(final_num_names),
                },
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        if final_cat_names is not None and len(final_cat_names) != X_cat_np.shape[1]:
            raise DataError(
                "Length of categorical_feature_names does not match number of categorical feature columns.",
                code="embedding_tabular_cat_feature_names_mismatch",
                context={
                    "num_categorical_features": int(X_cat_np.shape[1]),
                    "len_categorical_feature_names": len(final_cat_names),
                },
                location="ml_tabular.torch.datasets.tabular.EmbeddingTabularDataset.__init__",
            )

        # Convert to tensors
        self._X_num = torch.as_tensor(X_num_np, dtype=numeric_feature_dtype)
        self._X_cat = torch.as_tensor(X_cat_np, dtype=categorical_feature_dtype)

        if y_np is not None:
            if target_dtype is None:
                target_dtype = TabularDataset._infer_target_dtype(y_np)
            self._y = torch.as_tensor(y_np, dtype=target_dtype)
        else:
            self._y = None

        self._metadata = EmbeddingTabularDatasetMetadata(
            num_numeric_features=int(self._X_num.shape[1]),
            num_categorical_features=int(self._X_cat.shape[1]),
            numeric_feature_names=final_num_names,
            categorical_feature_names=final_cat_names,
            has_targets=self._y is not None,
        )

        logger.info(
            "Created EmbeddingTabularDataset: n_samples=%d, n_num_features=%d, "
            "n_cat_features=%d, has_targets=%s",
            self._X_num.shape[0],
            self._X_num.shape[1],
            self._X_cat.shape[1],
            self._metadata.has_targets,
        )

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._X_num.shape[0]

    def __getitem__(self, idx: int):
        x_num = self._X_num[idx]
        x_cat = self._X_cat[idx]

        if self._y is None:
            # Structure expected by generic training loops:
            #   inputs = (x_num, x_cat)
            return (x_num, x_cat)

        # With targets:
        #   ((x_num, x_cat), y)
        return (x_num, x_cat), self._y[idx]

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def numeric_features(self) -> torch.Tensor:
        """Return the numeric feature tensor X_num (shape: [n_samples, n_num_features])."""
        return self._X_num

    @property
    def categorical_features(self) -> torch.Tensor:
        """Return the categorical feature tensor X_cat (shape: [n_samples, n_cat_features])."""
        return self._X_cat

    @property
    def targets(self) -> Optional[torch.Tensor]:
        """Return the target tensor y (shape: [n_samples]) or None if not present."""
        return self._y

    @property
    def metadata(self) -> EmbeddingTabularDatasetMetadata:
        """Return metadata about this embedding-style dataset."""
        return self._metadata

    # ------------------------------------------------------------------
    # Alternative constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y: Optional[np.ndarray] = None,
        *,
        numeric_feature_dtype: torch.dtype = torch.float32,
        categorical_feature_dtype: torch.dtype = torch.long,
        target_dtype: Optional[torch.dtype] = None,
        numeric_feature_names: Optional[Sequence[str]] = None,
        categorical_feature_names: Optional[Sequence[str]] = None,
    ) -> "EmbeddingTabularDataset":
        """Construct an EmbeddingTabularDataset from numpy arrays.

        Parameters
        ----------
        X_num:
            Numeric feature matrix of shape (n_samples, n_num_features) or (n_samples,).
        X_cat:
            Categorical feature matrix of shape (n_samples, n_cat_features) or (n_samples,).
            Values are typically integer indices for embedding lookups.
        y:
            Optional target vector of shape (n_samples,) or (n_samples, 1).
        numeric_feature_dtype:
            torch.dtype for numeric features (default: float32).
        categorical_feature_dtype:
            torch.dtype for categorical features (default: long).
        target_dtype:
            Optional torch.dtype for targets; if None, it will be inferred.
        numeric_feature_names:
            Optional feature names for numeric features.
        categorical_feature_names:
            Optional feature names for categorical features.

        Returns
        -------
        EmbeddingTabularDataset
        """
        return cls(
            X_num=X_num,
            X_cat=X_cat,
            targets=y,
            numeric_feature_dtype=numeric_feature_dtype,
            categorical_feature_dtype=categorical_feature_dtype,
            target_dtype=target_dtype,
            numeric_feature_names=numeric_feature_names,
            categorical_feature_names=categorical_feature_names,
        )


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def make_embedding_tabular_dataset(
    X_num: Union[np.ndarray, pd.DataFrame],
    X_cat: Union[np.ndarray, pd.DataFrame],
    y: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
    *,
    numeric_feature_dtype: torch.dtype = torch.float32,
    categorical_feature_dtype: torch.dtype = torch.long,
    target_dtype: Optional[torch.dtype] = None,
    numeric_feature_names: Optional[Sequence[str]] = None,
    categorical_feature_names: Optional[Sequence[str]] = None,
) -> EmbeddingTabularDataset:
    """Factory function to build an EmbeddingTabularDataset.

    This keeps embedding-specific dataset logic in one well-known place,
    while letting callers stay terse in notebooks and scripts.

    Example
    -------
    >>> ds = make_embedding_tabular_dataset(
    ...     X_num=X_num,
    ...     X_cat=X_cat,
    ...     y=y,
    ... )
    >>> for (x_num, x_cat), y_batch in DataLoader(ds, batch_size=128, shuffle=True):
    ...     # x_num: (B, n_num_features)
    ...     # x_cat: (B, n_cat_features)
    ...     ...

    The returned dataset cooperates with the generic training loops that
    move nested structures of tensors to devices.
    """
    return EmbeddingTabularDataset(
        X_num=X_num,
        X_cat=X_cat,
        targets=y,
        numeric_feature_dtype=numeric_feature_dtype,
        categorical_feature_dtype=categorical_feature_dtype,
        target_dtype=target_dtype,
        numeric_feature_names=numeric_feature_names,
        categorical_feature_names=categorical_feature_names,
    )


__all__ = [
    "TabularDatasetMetadata",
    "TabularDataset",
    "EmbeddingTabularDatasetMetadata",
    "EmbeddingTabularDataset",
    "make_embedding_tabular_dataset",
]
