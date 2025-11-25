from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Metadata container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimeSeriesDatasetMetadata:
    """Metadata describing a time-series sliding-window dataset.

    Attributes
    ----------
    num_features:
        Number of input features per time step (D_in).
    num_targets:
        Number of target features per time step (D_out). Zero if no targets.
    input_window:
        Number of past time steps in each input window (history length).
    prediction_horizon:
        Number of future time steps to predict for each sample.
    step_size:
        Stride between successive window start positions in the original series.
    has_targets:
        Whether this dataset includes targets (y) for supervised learning.
    total_length:
        Total number of time steps in the underlying series (T).
    value_column_names:
        Optional feature column names (for DataFrame inputs).
    target_column_names:
        Optional target column names (for DataFrame inputs).
    """

    num_features: int
    num_targets: int
    input_window: int
    prediction_horizon: int
    step_size: int
    has_targets: bool
    total_length: int
    value_column_names: Optional[List[str]] = None
    target_column_names: Optional[List[str]] = None


# ---------------------------------------------------------------------------
# Core sliding-window Dataset
# ---------------------------------------------------------------------------


class TimeSeriesSequenceDataset(Dataset):
    """PyTorch Dataset for sliding-window time-series forecasting.

    This dataset takes a time-ordered sequence of feature vectors
    (and optionally target values) and generates overlapping windows:

        X[i] = features[t : t + input_window, :]
        y[i] = targets[t + input_window : t + input_window + prediction_horizon, :]

    for t in {0, step_size, 2 * step_size, ...} such that the full window and
    horizon lie within the series.

    Shapes
    ------
    Let:
        T = number of time steps
        D_in = number of input features
        D_out = number of target features
        W = input_window
        H = prediction_horizon

    Then each sample is:
        X_window:  (W, D_in)
        y_window:  (H, D_out)  # if targets are present

    And the dataset length is:
        N = floor((T - W - H) / step_size) + 1   (if T >= W + H, else error)

    Typical usage
    -------------
    From numpy arrays:

        series: np.ndarray  # shape (T, D_in)
        targets: np.ndarray # shape (T,) or (T, D_out)   (optional)

        ds = TimeSeriesSequenceDataset(
            series=series,
            targets=targets,
            input_window=30,
            prediction_horizon=7,
            step_size=1,
        )

    From a pandas DataFrame:

        df = ...
        ds = TimeSeriesSequenceDataset.from_dataframe(
            df,
            value_columns=["temp", "humidity"],
            target_column="temp",
            input_window=30,
            prediction_horizon=7,
            sort_by="timestamp",
        )

    In your training loop:

        loader = DataLoader(ds, batch_size=32, shuffle=True)

        for X_batch, y_batch in loader:
            # X_batch: (B, W, D_in)
            # y_batch: (B, H, D_out)
            ...

    Notes
    -----
    - If targets are omitted, __getitem__ returns only X_window (no y).
    - Normalization/standardization should be done in a separate feature layer;
      this dataset assumes you already have numeric features prepared.
    """

    def __init__(
        self,
        series: Union[np.ndarray, pd.DataFrame],
        targets: Optional[Union[np.ndarray, pd.Series, pd.DataFrame]] = None,
        *,
        input_window: int,
        prediction_horizon: int = 1,
        step_size: int = 1,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
        value_column_names: Optional[Sequence[str]] = None,
        target_column_names: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()

        if input_window <= 0:
            raise DataError(
                "input_window must be a positive integer.",
                code="ts_invalid_input_window",
                context={"input_window": input_window},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
            )
        if prediction_horizon <= 0:
            raise DataError(
                "prediction_horizon must be a positive integer.",
                code="ts_invalid_prediction_horizon",
                context={"prediction_horizon": prediction_horizon},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
            )
        if step_size <= 0:
            raise DataError(
                "step_size must be a positive integer.",
                code="ts_invalid_step_size",
                context={"step_size": step_size},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
            )

        series_np, inferred_value_names = self._series_to_2d_numpy(series)
        T, D_in = series_np.shape

        targets_np: Optional[np.ndarray]
        inferred_target_names: Optional[List[str]]
        if targets is not None:
            targets_np, inferred_target_names = self._targets_to_2d_numpy(targets, expected_length=T)
        else:
            targets_np, inferred_target_names = None, None

        # Use explicit column name lists if provided; otherwise use inferred from DataFrame
        final_value_names: Optional[List[str]]
        if value_column_names is not None:
            final_value_names = list(value_column_names)
            if len(final_value_names) != D_in:
                raise DataError(
                    "Length of value_column_names does not match number of input features.",
                    code="ts_value_names_mismatch",
                    context={"num_features": D_in, "len_value_column_names": len(final_value_names)},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
                )
        else:
            final_value_names = inferred_value_names

        if targets_np is not None:
            D_out = targets_np.shape[1]
        else:
            D_out = 0

        final_target_names: Optional[List[str]]
        if target_column_names is not None:
            final_target_names = list(target_column_names)
            if D_out > 0 and len(final_target_names) != D_out:
                raise DataError(
                    "Length of target_column_names does not match number of target features.",
                    code="ts_target_names_mismatch",
                    context={"num_targets": D_out, "len_target_column_names": len(final_target_names)},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
                )
        else:
            final_target_names = inferred_target_names

        # Compute valid starting indices for sliding windows
        last_start = T - input_window - prediction_horizon
        if last_start < 0:
            raise DataError(
                "Time series is too short for the requested input_window and prediction_horizon.",
                code="ts_series_too_short",
                context={
                    "total_length": T,
                    "input_window": input_window,
                    "prediction_horizon": prediction_horizon,
                },
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
            )

        start_indices = list(range(0, last_start + 1, step_size))
        if not start_indices:
            raise DataError(
                "No valid window start positions could be computed.",
                code="ts_no_windows",
                context={
                    "total_length": T,
                    "input_window": input_window,
                    "prediction_horizon": prediction_horizon,
                    "step_size": step_size,
                },
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.__init__",
            )

        self._series = torch.as_tensor(series_np, dtype=feature_dtype)  # (T, D_in)
        self._targets = (
            torch.as_tensor(targets_np, dtype=(target_dtype or self._infer_target_dtype(targets_np)))
            if targets_np is not None
            else None
        )
        self._start_indices = np.asarray(start_indices, dtype=np.int64)

        self._input_window = input_window
        self._prediction_horizon = prediction_horizon
        self._step_size = step_size

        self._metadata = TimeSeriesDatasetMetadata(
            num_features=D_in,
            num_targets=D_out,
            input_window=input_window,
            prediction_horizon=prediction_horizon,
            step_size=step_size,
            has_targets=self._targets is not None,
            total_length=T,
            value_column_names=final_value_names,
            target_column_names=final_target_names,
        )

        logger.info(
            "Created TimeSeriesSequenceDataset: T=%d, D_in=%d, D_out=%d, input_window=%d, "
            "prediction_horizon=%d, step_size=%d, n_samples=%d, has_targets=%s",
            T,
            D_in,
            D_out,
            input_window,
            prediction_horizon,
            step_size,
            len(self._start_indices),
            self._metadata.has_targets,
        )

    # ------------------------------------------------------------------
    # Internal conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _series_to_2d_numpy(
        series: Union[np.ndarray, pd.DataFrame],
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Convert time-series features to a 2D numpy array (T, D_in) and optional names."""
        if isinstance(series, pd.DataFrame):
            if series.empty:
                raise DataError(
                    "Input feature DataFrame for time series is empty.",
                    code="ts_series_df_empty",
                    context={},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._series_to_2d_numpy",
                )
            value_names = list(series.columns)
            arr = series.to_numpy()
        elif isinstance(series, np.ndarray):
            if series.size == 0:
                raise DataError(
                    "Input feature array for time series is empty.",
                    code="ts_series_array_empty",
                    context={},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._series_to_2d_numpy",
                )
            if series.ndim == 1:
                arr = series.reshape(-1, 1)
            elif series.ndim == 2:
                arr = series
            else:
                raise DataError(
                    "Time-series feature array must be 1D or 2D.",
                    code="ts_series_array_bad_ndim",
                    context={"shape": series.shape},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._series_to_2d_numpy",
                )
            value_names = None
        else:
            raise DataError(
                "Unsupported type for time-series features; expected numpy.ndarray or pandas.DataFrame.",
                code="ts_series_bad_type",
                context={"type": type(series).__name__},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._series_to_2d_numpy",
            )

        arr = np.asarray(arr)
        if arr.ndim != 2:
            raise DataError(
                "Time-series feature array must be 2D after conversion.",
                code="ts_series_not_2d",
                context={"shape": arr.shape},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._series_to_2d_numpy",
            )

        return arr, value_names

    @staticmethod
    def _targets_to_2d_numpy(
        targets: Union[np.ndarray, pd.Series, pd.DataFrame],
        expected_length: int,
    ) -> Tuple[np.ndarray, Optional[List[str]]]:
        """Convert targets to a 2D numpy array (T, D_out) and optional names."""
        if isinstance(targets, pd.DataFrame):
            if targets.empty:
                raise DataError(
                    "Target DataFrame for time series is empty.",
                    code="ts_targets_df_empty",
                    context={},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._targets_to_2d_numpy",
                )
            arr = targets.to_numpy()
            target_names = list(targets.columns)
        elif isinstance(targets, pd.Series):
            arr = targets.to_numpy().reshape(-1, 1)
            target_names = [targets.name] if targets.name is not None else None
        elif isinstance(targets, np.ndarray):
            if targets.size == 0:
                raise DataError(
                    "Target array for time series is empty.",
                    code="ts_targets_array_empty",
                    context={},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._targets_to_2d_numpy",
                )
            if targets.ndim == 1:
                arr = targets.reshape(-1, 1)
            elif targets.ndim == 2:
                arr = targets
            else:
                raise DataError(
                    "Time-series target array must be 1D or 2D.",
                    code="ts_targets_array_bad_ndim",
                    context={"shape": targets.shape},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._targets_to_2d_numpy",
                )
            target_names = None
        else:
            raise DataError(
                "Unsupported type for time-series targets; expected numpy.ndarray, pandas.Series, or pandas.DataFrame.",
                code="ts_targets_bad_type",
                context={"type": type(targets).__name__},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._targets_to_2d_numpy",
            )

        arr = np.asarray(arr)
        if arr.shape[0] != expected_length:
            raise DataError(
                "Targets and features must have the same number of time steps.",
                code="ts_targets_length_mismatch",
                context={"targets_length": arr.shape[0], "expected_length": expected_length},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset._targets_to_2d_numpy",
            )

        return arr, target_names

    @staticmethod
    def _infer_target_dtype(y_np: np.ndarray) -> torch.dtype:
        """Infer a reasonable torch dtype for targets based on their numpy dtype."""
        if np.issubdtype(y_np.dtype, np.integer):
            return torch.long
        if np.issubdtype(y_np.dtype, np.floating):
            return torch.float32
        logger.warning(
            "Time-series target dtype %s not clearly integer or float; defaulting to float32.",
            y_np.dtype,
        )
        return torch.float32

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return int(self._start_indices.shape[0])

    def __getitem__(self, idx: int):
        """Return the (X_window, y_window) pair for a given index.

        X_window shape: (input_window, num_features)
        y_window shape: (prediction_horizon, num_targets)

        If no targets are present, returns only X_window.
        """
        start = int(self._start_indices[idx])
        W = self._input_window
        H = self._prediction_horizon

        X_window = self._series[start : start + W, :]  # (W, D_in)

        if self._targets is None:
            return X_window

        y_start = start + W
        y_window = self._targets[y_start : y_start + H, :]  # (H, D_out)
        return X_window, y_window

    # ------------------------------------------------------------------
    # Public properties & alternative constructors
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> TimeSeriesDatasetMetadata:
        """Return metadata about this time-series dataset."""
        return self._metadata

    @property
    def input_window(self) -> int:
        return self._input_window

    @property
    def prediction_horizon(self) -> int:
        return self._prediction_horizon

    @property
    def step_size(self) -> int:
        return self._step_size

    @classmethod
    def from_arrays(
        cls,
        series: np.ndarray,
        targets: Optional[np.ndarray] = None,
        *,
        input_window: int,
        prediction_horizon: int = 1,
        step_size: int = 1,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
    ) -> "TimeSeriesSequenceDataset":
        """Construct a TimeSeriesSequenceDataset directly from numpy arrays.

        Parameters
        ----------
        series:
            Time-series feature array of shape (T, D_in) or (T,).
        targets:
            Optional target array of shape (T,), (T, 1), or (T, D_out).
        input_window:
            Number of past steps in each input window.
        prediction_horizon:
            Number of future steps to predict.
        step_size:
            Stride between successive window start positions.
        feature_dtype:
            torch.dtype for features (default: float32).
        target_dtype:
            Optional torch.dtype for targets; if None, it will be inferred.
        """
        return cls(
            series=series,
            targets=targets,
            input_window=input_window,
            prediction_horizon=prediction_horizon,
            step_size=step_size,
            feature_dtype=feature_dtype,
            target_dtype=target_dtype,
        )

    @classmethod
    def from_dataframe(
        cls,
        df: pd.DataFrame,
        *,
        value_columns: Optional[Sequence[str]] = None,
        target_column: Optional[str] = None,
        sort_by: Optional[str] = None,
        dropna: bool = True,
        input_window: int,
        prediction_horizon: int = 1,
        step_size: int = 1,
        feature_dtype: torch.dtype = torch.float32,
        target_dtype: Optional[torch.dtype] = None,
    ) -> "TimeSeriesSequenceDataset":
        """Construct a TimeSeriesSequenceDataset from a pandas DataFrame.

        Parameters
        ----------
        df:
            Input DataFrame containing time-ordered data. It is strongly
            recommended that df is sorted by time before calling this method
            or that you provide sort_by.
        value_columns:
            Names of columns to use as input features. If None, all columns
            except target_column (if provided) are used.
        target_column:
            Name of the column to use as the forecast target. If None, the
            dataset will be created without targets (unsupervised/inference).
        sort_by:
            Optional name of a column to sort by (e.g. a timestamp column).
            If provided, df will be sorted ascending by this column before
            constructing the dataset.
        dropna:
            If True, rows with NaNs in any used value/target columns are dropped.
        input_window:
            Number of past steps in each input window.
        prediction_horizon:
            Number of future steps to predict.
        step_size:
            Stride between successive window start positions.
        feature_dtype:
            torch.dtype for features (default: float32).
        target_dtype:
            Optional torch.dtype for targets; if None, it will be inferred.

        Returns
        -------
        TimeSeriesSequenceDataset

        Raises
        ------
        DataError
            If required columns are missing or if no value columns remain.
        """
        if df.empty:
            raise DataError(
                "Input DataFrame for time series is empty.",
                code="ts_df_empty",
                context={},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
            )

        df = df.copy()

        # Sort by time if requested
        if sort_by is not None:
            if sort_by not in df.columns:
                raise DataError(
                    f"sort_by column '{sort_by}' not found in DataFrame.",
                    code="ts_sort_by_missing",
                    context={"columns": list(df.columns)},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
                )
            df = df.sort_values(by=sort_by)

        # Determine value columns
        if value_columns is None:
            if target_column is not None:
                if target_column not in df.columns:
                    raise DataError(
                        f"Target column '{target_column}' not found in DataFrame.",
                        code="ts_target_missing",
                        context={"columns": list(df.columns)},
                        location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
                    )
                value_cols = [c for c in df.columns if c != target_column]
            else:
                value_cols = list(df.columns)
        else:
            missing_values = [c for c in value_columns if c not in df.columns]
            if missing_values:
                raise DataError(
                    "Some value_columns are not present in the DataFrame.",
                    code="ts_value_columns_missing",
                    context={"missing": missing_values, "columns": list(df.columns)},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
                )
            value_cols = list(value_columns)

        if not value_cols:
            raise DataError(
                "No value columns remain for time-series features.",
                code="ts_no_value_columns",
                context={"target_column": target_column},
                location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
            )

        # Build series and targets
        series_df = df[value_cols]

        if target_column is not None:
            if target_column not in df.columns:
                raise DataError(
                    f"Target column '{target_column}' not found in DataFrame.",
                    code="ts_target_missing",
                    context={"columns": list(df.columns)},
                    location="ml_tabular.torch.datasets.time_series.TimeSeriesSequenceDataset.from_dataframe",
                )
            targets_obj: Union[pd.Series, pd.DataFrame] = df[target_column]
        else:
            targets_obj = None  # type: ignore[assignment]

        if dropna:
            # Drop rows with NaNs in any used column (features + target if present)
            cols_to_check = list(value_cols)
            if target_column is not None:
                cols_to_check.append(target_column)
            before = len(df)
            df = df.dropna(subset=cols_to_check)
            series_df = df[value_cols]
            if target_column is not None:
                targets_obj = df[target_column]
            dropped = before - len(df)
            if dropped > 0:
                logger.info("Dropped %d rows with NaNs from time-series DataFrame.", dropped)

        return cls(
            series=series_df,
            targets=targets_obj,
            input_window=input_window,
            prediction_horizon=prediction_horizon,
            step_size=step_size,
            feature_dtype=feature_dtype,
            target_dtype=target_dtype,
            value_column_names=value_cols,
            target_column_names=[target_column] if target_column is not None else None,
        )
