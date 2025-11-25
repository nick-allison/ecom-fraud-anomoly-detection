from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)
_LOCATION_PREFIX = __name__


@dataclass(frozen=True)
class TimeSeriesFeatureSpec:
    """Specification for stateless time-series feature engineering.

    This is meant to be generic enough to cover common forecasting use-cases:
    - expanding a datetime column into calendar/time parts
    - creating lags, differences, and rolling-window aggregates
    - optionally respecting a per-series group column (e.g. store_id)

    All transformations are *stateless*: they operate directly on the input
    DataFrame and do not keep any learned parameters.
    """

    # Name of the datetime column (optional, but required for datetime expansion)
    datetime_column: str | None = None

    # Optional column indicating the series/group ID (e.g. "store_id").
    # If provided, lag, diff, and rolling features are computed within each group.
    group_column: str | None = None

    # Whether to sort by (group_column, datetime_column) before applying
    # lag/diff/rolling operations. Recommended for well-formed time-series data.
    sort_by_time: bool = True

    # If True and datetime_column is set, expand datetime into calendar features.
    expand_datetime: bool = False

    # Whether to drop the original datetime_column after expansion.
    drop_original_datetime: bool = False

    # Mapping from new feature name -> (source_column, lag_steps)
    lag_features: Mapping[str, Tuple[str, int]] = field(default_factory=dict)

    # Mapping from new feature name -> (source_column, lag_steps)
    diff_features: Mapping[str, Tuple[str, int]] = field(default_factory=dict)

    # Mapping from new feature name -> (source_column, window_size, agg)
    # where agg is one of {"mean", "sum", "min", "max", "std"}.
    rolling_features: Mapping[str, Tuple[str, int, str]] = field(default_factory=dict)

    # Minimum non-null observations required in a window to compute a rolling value.
    # If None, defaults to the full window size.
    rolling_min_periods: int | None = None


# Default spec does nothing; customize this in real projects.
DEFAULT_TS_FEATURE_SPEC = TimeSeriesFeatureSpec()


def build_time_series_features(
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
    spec: TimeSeriesFeatureSpec | None = None,
) -> pd.DataFrame:
    """Apply stateless time-series feature transformations to a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame. Typically contains at least a datetime column, a
        target column, and optional covariates. May also include a group ID
        column for multiple related series.
    dataset_name:
        Optional logical name (e.g. "train", "valid", "test") used in logs.
    spec:
        TimeSeriesFeatureSpec describing which transformations to apply.
        If None, DEFAULT_TS_FEATURE_SPEC is used.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with additional time-series feature columns added,
        and optionally the original datetime column dropped.
    """
    if spec is None:
        spec = DEFAULT_TS_FEATURE_SPEC

    logger.info(
        "Starting time-series feature engineering",
        extra={
            "dataset_name": dataset_name,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "datetime_column": spec.datetime_column,
            "group_column": spec.group_column,
            "sort_by_time": spec.sort_by_time,
            "expand_datetime": spec.expand_datetime,
            "drop_original_datetime": spec.drop_original_datetime,
            "n_lag_features": len(spec.lag_features),
            "n_diff_features": len(spec.diff_features),
            "n_rolling_features": len(spec.rolling_features),
        },
    )

    # Defensive copy so we never mutate the caller's DataFrame.
    features_df = df.copy()

    # Optional sort by (group, datetime) for well-defined lag/diff/rolling.
    if spec.sort_by_time and spec.datetime_column is not None:
        features_df = _sort_by_time(
            features_df,
            datetime_column=spec.datetime_column,
            group_column=spec.group_column,
            dataset_name=dataset_name,
        )

    # Optional datetime expansion into calendar/time parts.
    if spec.expand_datetime and spec.datetime_column is not None:
        features_df = _add_datetime_parts(
            features_df,
            datetime_column=spec.datetime_column,
            drop_original=spec.drop_original_datetime,
            dataset_name=dataset_name,
        )

    if spec.lag_features:
        features_df = _add_lag_features(
            features_df,
            lag_spec=spec.lag_features,
            group_column=spec.group_column,
            dataset_name=dataset_name,
        )

    if spec.diff_features:
        features_df = _add_diff_features(
            features_df,
            diff_spec=spec.diff_features,
            group_column=spec.group_column,
            dataset_name=dataset_name,
        )

    if spec.rolling_features:
        features_df = _add_rolling_features(
            features_df,
            rolling_spec=spec.rolling_features,
            min_periods=spec.rolling_min_periods,
            group_column=spec.group_column,
            dataset_name=dataset_name,
        )

    logger.info(
        "Finished time-series feature engineering",
        extra={
            "dataset_name": dataset_name,
            "n_rows": int(features_df.shape[0]),
            "n_cols": int(features_df.shape[1]),
        },
    )

    return features_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sort_by_time(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    group_column: str | None,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Sort the DataFrame by (group_column, datetime_column) or datetime only.

    This ensures lag/diff/rolling features follow the natural time order within
    each series.
    """
    if datetime_column not in df.columns:
        raise DataError(
            "Datetime column is missing from the dataframe",
            code="ts_missing_datetime_column",
            context={
                "dataset_name": dataset_name,
                "datetime_column": datetime_column,
            },
            location=f"{_LOCATION_PREFIX}._sort_by_time",
        )

    sort_cols: list[str] = []
    if group_column is not None:
        if group_column not in df.columns:
            raise DataError(
                "Group column is missing from the dataframe",
                code="ts_missing_group_column",
                context={
                    "dataset_name": dataset_name,
                    "group_column": group_column,
                },
                location=f"{_LOCATION_PREFIX}._sort_by_time",
            )
        sort_cols.append(group_column)

    sort_cols.append(datetime_column)

    try:
        sorted_df = df.sort_values(sort_cols).reset_index(drop=True)
    except Exception as exc:
        raise DataError(
            "Failed to sort dataframe by time",
            code="ts_sort_error",
            cause=exc,
            context={"dataset_name": dataset_name, "sort_columns": sort_cols},
            location=f"{_LOCATION_PREFIX}._sort_by_time",
        ) from exc

    return sorted_df


def _add_datetime_parts(
    df: pd.DataFrame,
    *,
    datetime_column: str,
    drop_original: bool,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Expand a datetime column into standard calendar/time features.

    For datetime column `col`, creates:

      - {col}__year
      - {col}__month
      - {col}__day
      - {col}__dayofweek
      - {col}__hour
      - {col}__weekofyear
      - {col}__is_weekend
      - {col}__is_month_start
      - {col}__is_month_end
    """
    if datetime_column not in df.columns:
        raise DataError(
            "Datetime column is missing from the dataframe",
            code="ts_missing_datetime_column",
            context={
                "dataset_name": dataset_name,
                "datetime_column": datetime_column,
            },
            location=f"{_LOCATION_PREFIX}._add_datetime_parts",
        )

    result = df

    series = result[datetime_column]
    try:
        dt = pd.to_datetime(series, errors="coerce")
    except Exception as exc:
        raise DataError(
            f"Failed to convert column '{datetime_column}' to datetime",
            code="ts_datetime_conversion_error",
            cause=exc,
            context={"dataset_name": dataset_name, "column": datetime_column},
            location=f"{_LOCATION_PREFIX}._add_datetime_parts",
        ) from exc

    n_invalid = int(dt.isna().sum())
    if n_invalid > 0:
        logger.warning(
            "Datetime conversion produced NaT values in time-series features",
            extra={
                "dataset_name": dataset_name,
                "column": datetime_column,
                "n_invalid": n_invalid,
            },
        )

    base = datetime_column

    result[f"{base}__year"] = dt.dt.year
    result[f"{base}__month"] = dt.dt.month
    result[f"{base}__day"] = dt.dt.day
    result[f"{base}__dayofweek"] = dt.dt.dayofweek
    result[f"{base}__hour"] = dt.dt.hour
    result[f"{base}__weekofyear"] = dt.dt.isocalendar().week.astype("Int64")
    result[f"{base}__is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")
    result[f"{base}__is_month_start"] = dt.dt.is_month_start.astype("Int64")
    result[f"{base}__is_month_end"] = dt.dt.is_month_end.astype("Int64")

    if drop_original:
        result = result.drop(columns=[datetime_column])

    return result


def _add_lag_features(
    df: pd.DataFrame,
    *,
    lag_spec: Mapping[str, Tuple[str, int]],
    group_column: str | None,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create lag features: new_col = source_column shifted by lag steps.

    Lags are computed within each group if `group_column` is provided; otherwise
    over the full series.
    """
    result = df

    for new_col, (src_col, lag) in lag_spec.items():
        if src_col not in result.columns:
            raise DataError(
                "Missing source column for lag feature",
                code="ts_missing_lag_source",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "lag": lag,
                },
                location=f"{_LOCATION_PREFIX}._add_lag_features",
            )

        if lag <= 0:
            raise DataError(
                "Lag must be a positive integer",
                code="ts_invalid_lag",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "lag": lag,
                },
                location=f"{_LOCATION_PREFIX}._add_lag_features",
            )

        if group_column is not None:
            if group_column not in result.columns:
                raise DataError(
                    "Group column is missing for lag feature computation",
                    code="ts_missing_group_column",
                    context={
                        "dataset_name": dataset_name,
                        "group_column": group_column,
                        "new_column": new_col,
                    },
                    location=f"{_LOCATION_PREFIX}._add_lag_features",
                )
            result[new_col] = (
                result.groupby(group_column)[src_col].shift(lag)
            )
        else:
            result[new_col] = result[src_col].shift(lag)

    return result


def _add_diff_features(
    df: pd.DataFrame,
    *,
    diff_spec: Mapping[str, Tuple[str, int]],
    group_column: str | None,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create difference features: new_col = source_column.diff(lag)."""
    result = df

    for new_col, (src_col, lag) in diff_spec.items():
        if src_col not in result.columns:
            raise DataError(
                "Missing source column for diff feature",
                code="ts_missing_diff_source",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "lag": lag,
                },
                location=f"{_LOCATION_PREFIX}._add_diff_features",
            )

        if lag <= 0:
            raise DataError(
                "Diff lag must be a positive integer",
                code="ts_invalid_diff_lag",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "lag": lag,
                },
                location=f"{_LOCATION_PREFIX}._add_diff_features",
            )

        if group_column is not None:
            if group_column not in result.columns:
                raise DataError(
                    "Group column is missing for diff feature computation",
                    code="ts_missing_group_column",
                    context={
                        "dataset_name": dataset_name,
                        "group_column": group_column,
                        "new_column": new_col,
                    },
                    location=f"{_LOCATION_PREFIX}._add_diff_features",
                )
            result[new_col] = result.groupby(group_column)[src_col].diff(lag)
        else:
            result[new_col] = result[src_col].diff(lag)

    return result


def _add_rolling_features(
    df: pd.DataFrame,
    *,
    rolling_spec: Mapping[str, Tuple[str, int, str]],
    min_periods: int | None,
    group_column: str | None,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create rolling-window features.

    For each entry in `rolling_spec`:
        new_col -> (source_column, window_size, agg)

    where agg is one of: {"mean", "sum", "min", "max", "std"}.

    Rolling windows are computed within each group if `group_column` is
    provided; otherwise over the full series.
    """
    result = df

    valid_aggs = {"mean", "sum", "min", "max", "std"}

    for new_col, (src_col, window, agg) in rolling_spec.items():
        if src_col not in result.columns:
            raise DataError(
                "Missing source column for rolling feature",
                code="ts_missing_rolling_source",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "window": window,
                    "agg": agg,
                },
                location=f"{_LOCATION_PREFIX}._add_rolling_features",
            )

        if window <= 0:
            raise DataError(
                "Rolling window size must be a positive integer",
                code="ts_invalid_rolling_window",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "window": window,
                    "agg": agg,
                },
                location=f"{_LOCATION_PREFIX}._add_rolling_features",
            )

        if agg not in valid_aggs:
            raise DataError(
                "Invalid rolling aggregation function",
                code="ts_invalid_rolling_agg",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "window": window,
                    "agg": agg,
                    "valid_aggs": sorted(valid_aggs),
                },
                location=f"{_LOCATION_PREFIX}._add_rolling_features",
            )

        effective_min_periods = window if min_periods is None else min_periods

        # Convert to numeric; non-numeric values become NaN.
        numeric = pd.to_numeric(result[src_col], errors="coerce")
        n_non_numeric = int(numeric.isna().sum() - result[src_col].isna().sum())
        if n_non_numeric > 0:
            logger.warning(
                "Non-numeric values encountered when creating rolling feature",
                extra={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "window": window,
                    "agg": agg,
                    "n_non_numeric": n_non_numeric,
                },
            )

        if group_column is not None:
            if group_column not in result.columns:
                raise DataError(
                    "Group column is missing for rolling feature computation",
                    code="ts_missing_group_column",
                    context={
                        "dataset_name": dataset_name,
                        "group_column": group_column,
                        "new_column": new_col,
                    },
                    location=f"{_LOCATION_PREFIX}._add_rolling_features",
                )

            grouped = numeric.groupby(result[group_column], sort=False)
            if agg == "mean":
                rolled = grouped.transform(
                    lambda s: s.rolling(window, min_periods=effective_min_periods).mean()
                )
            elif agg == "sum":
                rolled = grouped.transform(
                    lambda s: s.rolling(window, min_periods=effective_min_periods).sum()
                )
            elif agg == "min":
                rolled = grouped.transform(
                    lambda s: s.rolling(window, min_periods=effective_min_periods).min()
                )
            elif agg == "max":
                rolled = grouped.transform(
                    lambda s: s.rolling(window, min_periods=effective_min_periods).max()
                )
            else:  # "std"
                rolled = grouped.transform(
                    lambda s: s.rolling(window, min_periods=effective_min_periods).std()
                )
        else:
            if agg == "mean":
                rolled = numeric.rolling(window, min_periods=effective_min_periods).mean()
            elif agg == "sum":
                rolled = numeric.rolling(window, min_periods=effective_min_periods).sum()
            elif agg == "min":
                rolled = numeric.rolling(window, min_periods=effective_min_periods).min()
            elif agg == "max":
                rolled = numeric.rolling(window, min_periods=effective_min_periods).max()
            else:  # "std"
                rolled = numeric.rolling(window, min_periods=effective_min_periods).std()

        result[new_col] = rolled

    return result
