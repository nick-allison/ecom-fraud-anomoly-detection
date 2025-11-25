from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence, Tuple, Mapping

import numpy as np
import pandas as pd

from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)
_LOCATION_PREFIX = __name__


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for stateless feature engineering.

    This is intentionally simple and meant to be customized per project.
    You can define a default spec in this module and override it in
    project-specific code or via configuration.
    """

    # Date/time columns to expand into year/month/day/day_of_week/hour
    datetime_columns: Sequence[str] = ()

    # Columns for which to create a log1p-transformed version
    log1p_columns: Sequence[str] = ()

    # Mapping from new feature name -> (numerator_column, denominator_column)
    ratio_features: Mapping[str, Tuple[str, str]] = field(default_factory=dict)

    # Mapping from new feature name -> (source_column, exponent)
    power_features: Mapping[str, Tuple[str, float]] = field(default_factory=dict)

    # Mapping from new feature name -> (column1, column2) to multiply
    interaction_features: Mapping[str, Tuple[str, str]] = field(default_factory=dict)

    # Whether to drop the original datetime columns after expansion
    drop_original_datetime: bool = False


# Default spec does nothing; customize this in real projects.
DEFAULT_FEATURE_SPEC = FeatureSpec(
    datetime_columns=(),
    log1p_columns=(),
    ratio_features={},
    power_features={},
    interaction_features={},
    drop_original_datetime=False,
)


def build_features(
    df: pd.DataFrame,
    *,
    dataset_name: str | None = None,
    spec: FeatureSpec | None = None,
) -> pd.DataFrame:
    """Apply stateless feature transformations to a DataFrame.

    Parameters
    ----------
    df:
        Input DataFrame. Should already be loaded and structurally validated.
    dataset_name:
        Optional logical name (e.g. 'train', 'valid', 'test') used in logs/errors.
    spec:
        FeatureSpec describing which transformations to apply. If None,
        DEFAULT_FEATURE_SPEC is used.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with additional feature columns (and optionally
        some columns dropped, e.g. datetime originals).
    """
    if spec is None:
        spec = DEFAULT_FEATURE_SPEC

    logger.info(
        "Starting feature engineering",
        extra={
            "dataset_name": dataset_name,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
            "datetime_columns": list(spec.datetime_columns),
            "log1p_columns": list(spec.log1p_columns),
            "ratio_features": dict(spec.ratio_features),
            "power_features": dict(spec.power_features),
            "interaction_features": dict(spec.interaction_features),
            "drop_original_datetime": spec.drop_original_datetime,
        },
    )

    # Make a single defensive copy so we never mutate the caller's DataFrame.
    features_df = df.copy()

    if spec.datetime_columns:
        features_df = _add_datetime_features(
            features_df,
            columns=spec.datetime_columns,
            drop_original=spec.drop_original_datetime,
            dataset_name=dataset_name,
        )

    if spec.log1p_columns:
        features_df = _add_log1p_features(
            features_df,
            columns=spec.log1p_columns,
            dataset_name=dataset_name,
        )

    if spec.ratio_features:
        features_df = _add_ratio_features(
            features_df,
            ratio_spec=spec.ratio_features,
            dataset_name=dataset_name,
        )

    if spec.power_features:
        features_df = _add_power_features(
            features_df,
            power_spec=spec.power_features,
            dataset_name=dataset_name,
        )

    if spec.interaction_features:
        features_df = _add_interaction_features(
            features_df,
            interaction_spec=spec.interaction_features,
            dataset_name=dataset_name,
        )

    logger.info(
        "Finished feature engineering",
        extra={
            "dataset_name": dataset_name,
            "n_rows": int(features_df.shape[0]),
            "n_cols": int(features_df.shape[1]),
        },
    )

    return features_df


def _add_datetime_features(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    drop_original: bool,
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Expand datetime columns into standard numeric and boolean features.

    For each datetime column `col`, this creates:

      - {col}__year           : calendar year
      - {col}__month          : month number (1–12)
      - {col}__day            : day of month (1–31)
      - {col}__dayofweek      : day of week (0 = Monday, 6 = Sunday)
      - {col}__hour           : hour of day (0–23)
      - {col}__weekofyear     : ISO week number (1–53)
      - {col}__is_weekend     : 1 if Saturday/Sunday, else 0
      - {col}__is_month_start : 1 if first day of month, else 0
      - {col}__is_month_end   : 1 if last day of month, else 0

    Original columns can optionally be dropped (drop_original=True).
    """
    result = df  # operate on the already-copied frame

    missing_cols = [col for col in columns if col not in result.columns]
    if missing_cols:
        raise DataError(
            "One or more datetime columns are missing from the dataframe",
            code="feature_missing_datetime_columns",
            context={
                "dataset_name": dataset_name,
                "columns": list(columns),
                "missing_columns": missing_cols,
            },
            location=f"{_LOCATION_PREFIX}._add_datetime_features",
        )

    for col in columns:
        series = result[col]

        try:
            dt = pd.to_datetime(series, errors="coerce")
        except Exception as exc:
            raise DataError(
                f"Failed to convert column '{col}' to datetime",
                code="feature_datetime_conversion_error",
                cause=exc,
                context={"dataset_name": dataset_name, "column": col},
                location=f"{_LOCATION_PREFIX}._add_datetime_features",
            ) from exc

        n_invalid = int(dt.isna().sum())
        if n_invalid > 0:
            logger.warning(
                "Datetime conversion produced NaT values",
                extra={
                    "dataset_name": dataset_name,
                    "column": col,
                    "n_invalid": n_invalid,
                },
            )

        base_name = col

        # Basic calendar/time fields
        result[f"{base_name}__year"] = dt.dt.year
        result[f"{base_name}__month"] = dt.dt.month
        result[f"{base_name}__day"] = dt.dt.day
        result[f"{base_name}__dayofweek"] = dt.dt.dayofweek
        result[f"{base_name}__hour"] = dt.dt.hour

        # Richer temporal features
        # ISO week of year (1–53)
        result[f"{base_name}__weekofyear"] = dt.dt.isocalendar().week.astype("Int64")

        # Weekend indicator: Saturday (5) or Sunday (6)
        result[f"{base_name}__is_weekend"] = (dt.dt.dayofweek >= 5).astype("Int64")

        # Month boundary indicators
        result[f"{base_name}__is_month_start"] = dt.dt.is_month_start.astype("Int64")
        result[f"{base_name}__is_month_end"] = dt.dt.is_month_end.astype("Int64")

    if drop_original:
        result = result.drop(columns=list(columns))

    return result


def _add_log1p_features(
    df: pd.DataFrame,
    *,
    columns: Iterable[str],
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Add log1p-transformed versions of numeric columns.

    For each `col` in `columns`, this creates a new column named
    `{col}__log1p`.

    Notes
    -----
    - This transformation is typically used for count-like, non-negative
      features. If negative values are present, a DataError is raised.
    """
    result = df

    missing_cols = [col for col in columns if col not in result.columns]
    if missing_cols:
        raise DataError(
            "One or more log1p columns are missing from the dataframe",
            code="feature_missing_log1p_columns",
            context={
                "dataset_name": dataset_name,
                "columns": list(columns),
                "missing_columns": missing_cols,
            },
            location=f"{_LOCATION_PREFIX}._add_log1p_features",
        )

    for col in columns:
        series = result[col]

        # Ensure numeric type (or coercible to numeric).
        numeric = pd.to_numeric(series, errors="coerce")
        n_non_numeric = int(numeric.isna().sum() - series.isna().sum())
        if n_non_numeric > 0:
            raise DataError(
                "Non-numeric values found in log1p feature column",
                code="feature_log1p_non_numeric",
                context={
                    "dataset_name": dataset_name,
                    "column": col,
                    "n_non_numeric": n_non_numeric,
                },
                location=f"{_LOCATION_PREFIX}._add_log1p_features",
            )

        # Enforce non-negative domain for log1p-transform.
        n_negative = int((numeric < 0).sum())
        if n_negative > 0:
            raise DataError(
                "Negative values found in log1p feature column",
                code="feature_log1p_negative_values",
                context={
                    "dataset_name": dataset_name,
                    "column": col,
                    "n_negative": n_negative,
                },
                location=f"{_LOCATION_PREFIX}._add_log1p_features",
            )

        result[f"{col}__log1p"] = np.log1p(numeric)

    return result


def _add_ratio_features(
    df: pd.DataFrame,
    *,
    ratio_spec: Mapping[str, Tuple[str, str]],
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create ratio features defined as new_col = numerator / denominator."""
    result = df

    for new_col, (num_col, den_col) in ratio_spec.items():
        missing_cols = [c for c in (num_col, den_col) if c not in result.columns]
        if missing_cols:
            raise DataError(
                "Missing columns for ratio feature",
                code="feature_missing_ratio_columns",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "numerator": num_col,
                    "denominator": den_col,
                    "missing_columns": missing_cols,
                },
                location=f"{_LOCATION_PREFIX}._add_ratio_features",
            )

        num = pd.to_numeric(result[num_col], errors="coerce")
        den = pd.to_numeric(result[den_col], errors="coerce")

        # Track invalid numeric conversions
        n_num_invalid = int(num.isna().sum() - result[num_col].isna().sum())
        n_den_invalid = int(den.isna().sum() - result[den_col].isna().sum())
        if n_num_invalid > 0 or n_den_invalid > 0:
            logger.warning(
                "Non-numeric values encountered in ratio feature columns",
                extra={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "numerator": num_col,
                    "denominator": den_col,
                    "n_num_non_numeric": n_num_invalid,
                    "n_den_non_numeric": n_den_invalid,
                },
            )

        # Avoid division by zero: where denominator is 0, set ratio to NaN and log.
        zero_den = (den == 0) & den.notna()
        n_zero_den = int(zero_den.sum())
        if n_zero_den > 0:
            logger.warning(
                "Zero denominator encountered when creating ratio feature",
                extra={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "numerator": num_col,
                    "denominator": den_col,
                    "n_zero_denominator": n_zero_den,
                },
            )
            safe_den = den.mask(zero_den, np.nan)
        else:
            safe_den = den

        result[new_col] = num / safe_den

    return result


def _add_power_features(
    df: pd.DataFrame,
    *,
    power_spec: Mapping[str, Tuple[str, float]],
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create power features defined as new_col = source_column ** exponent."""
    result = df

    for new_col, (src_col, exponent) in power_spec.items():
        if src_col not in result.columns:
            raise DataError(
                "Missing source column for power feature",
                code="feature_missing_power_source",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "exponent": exponent,
                },
                location=f"{_LOCATION_PREFIX}._add_power_features",
            )

        numeric = pd.to_numeric(result[src_col], errors="coerce")
        n_non_numeric = int(numeric.isna().sum() - result[src_col].isna().sum())
        if n_non_numeric > 0:
            logger.warning(
                "Non-numeric values encountered when creating power feature",
                extra={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "source_column": src_col,
                    "exponent": exponent,
                    "n_non_numeric": n_non_numeric,
                },
            )

        result[new_col] = numeric**exponent

    return result


def _add_interaction_features(
    df: pd.DataFrame,
    *,
    interaction_spec: Mapping[str, Tuple[str, str]],
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """Create interaction features defined as new_col = col1 * col2."""
    result = df

    for new_col, (col1, col2) in interaction_spec.items():
        missing = [c for c in (col1, col2) if c not in result.columns]
        if missing:
            raise DataError(
                "Missing columns for interaction feature",
                code="feature_missing_interaction_columns",
                context={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "col1": col1,
                    "col2": col2,
                    "missing_columns": missing,
                },
                location=f"{_LOCATION_PREFIX}._add_interaction_features",
            )

        num1 = pd.to_numeric(result[col1], errors="coerce")
        num2 = pd.to_numeric(result[col2], errors="coerce")

        n_num1_invalid = int(num1.isna().sum() - result[col1].isna().sum())
        n_num2_invalid = int(num2.isna().sum() - result[col2].isna().sum())
        if n_num1_invalid > 0 or n_num2_invalid > 0:
            logger.warning(
                "Non-numeric values encountered when creating interaction feature",
                extra={
                    "dataset_name": dataset_name,
                    "new_column": new_col,
                    "col1": col1,
                    "col2": col2,
                    "n_col1_non_numeric": n_num1_invalid,
                    "n_col2_non_numeric": n_num2_invalid,
                },
            )

        result[new_col] = num1 * num2

    return result
