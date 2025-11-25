from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TabularPipelineConfig:
    """Configuration for a tabular preprocessing pipeline.

    This is a thin configuration layer around a sklearn ColumnTransformer
    that handles:

    - Numeric features:
        * Missing-value imputation (median/mean/most_frequent/constant)
        * Optional scaling (StandardScaler)

    - Categorical features:
        * Missing-value imputation (most_frequent/constant)
        * One-hot encoding, with configurable handling of unknown categories

    Feature sets can be specified explicitly via numeric_features,
    categorical_features and drop_features, or inferred from the DataFrame
    dtypes when not provided.

    Typical usage:

        cfg = TabularPipelineConfig()
        preprocessor = build_tabular_preprocessor(df, cfg)
        X_train_processed = preprocessor.fit_transform(df)

    You can then pass the resulting numpy arrays into PyTorch datasets or
    classical ML models.
    """

    # Explicit feature lists (optional). If None, they will be inferred from dtypes.
    numeric_features: Optional[Sequence[str]] = None
    categorical_features: Optional[Sequence[str]] = None
    drop_features: Sequence[str] = field(default_factory=list)

    # Numeric column behavior
    numeric_imputer_strategy: Literal["mean", "median", "most_frequent", "constant"] = "median"
    numeric_imputer_fill_value: Optional[float] = None
    scale_numeric: bool = True

    # Categorical column behavior
    categorical_imputer_strategy: Literal["most_frequent", "constant"] = "most_frequent"
    categorical_imputer_fill_value: Optional[str] = None
    ohe_handle_unknown: Literal["ignore", "error"] = "ignore"
    ohe_min_frequency: Optional[int] = None
    ohe_max_categories: Optional[int] = None

    # How to treat any columns not assigned to numeric/categorical
    remainder: Literal["drop", "passthrough"] = "drop"


# ---------------------------------------------------------------------------
# Feature inference helpers
# ---------------------------------------------------------------------------


def _infer_feature_types(
    df: pd.DataFrame,
    config: TabularPipelineConfig,
) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical features from a DataFrame and config.

    Rules:
      - If config.numeric_features is provided, use those (and validate they exist).
      - Otherwise, numeric = columns with "number" dtype.
      - If config.categorical_features is provided, use those (and validate).
      - Otherwise, categorical = object/category/bool columns.
      - Any columns in config.drop_features are removed from both sets.
      - Columns not in numeric/categorical/drop are treated according to config.remainder.

    Raises:
      DataError if there are no features at all after inference.
    """
    all_columns = set(df.columns)

    # Validate drop_features existence (warn but do not fail if some are missing)
    missing_drops = [c for c in config.drop_features if c not in all_columns]
    if missing_drops:
        logger.warning(
            "Some drop_features are not present in the DataFrame and will be ignored: %s",
            ", ".join(missing_drops),
        )

    drop_set = set(config.drop_features)

    # Numeric features
    if config.numeric_features is not None:
        missing_numeric = [c for c in config.numeric_features if c not in all_columns]
        if missing_numeric:
            raise DataError(
                "Some configured numeric_features are not present in the DataFrame.",
                code="missing_numeric_features",
                context={
                    "missing": missing_numeric,
                    "columns": list(df.columns),
                },
                location="ml_tabular.sklearn_utils.tabular_pipeline._infer_feature_types",
            )
        numeric = [c for c in config.numeric_features if c not in drop_set]
    else:
        numeric = [
            c
            for c in df.select_dtypes(include=["number"]).columns
            if c not in drop_set
        ]

    # Categorical features
    if config.categorical_features is not None:
        missing_cat = [c for c in config.categorical_features if c not in all_columns]
        if missing_cat:
            raise DataError(
                "Some configured categorical_features are not present in the DataFrame.",
                code="missing_categorical_features",
                context={
                    "missing": missing_cat,
                    "columns": list(df.columns),
                },
                location="ml_tabular.sklearn_utils.tabular_pipeline._infer_feature_types",
            )
        categorical = [c for c in config.categorical_features if c not in drop_set]
    else:
        categorical = [
            c
            for c in df.select_dtypes(include=["object", "category", "bool"]).columns
            if c not in drop_set
        ]

    # Safety: remove any overlap (if user accidentally specifies a column in both)
    overlap = set(numeric) & set(categorical)
    if overlap:
        logger.warning(
            "Columns specified as both numeric and categorical; treating them as numeric: %s",
            ", ".join(sorted(overlap)),
        )
        categorical = [c for c in categorical if c not in overlap]

    if not numeric and not categorical and config.remainder == "drop":
        raise DataError(
            "No features remaining after inferring numeric/categorical and dropping requested columns.",
            code="no_features_after_inference",
            context={
                "drop_features": list(drop_set),
                "columns": list(df.columns),
            },
            location="ml_tabular.sklearn_utils.tabular_pipeline._infer_feature_types",
        )

    logger.info(
        "Inferred feature types: numeric=%s, categorical=%s, dropped=%s",
        numeric,
        categorical,
        list(drop_set),
    )

    return numeric, categorical


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def _build_numeric_pipeline(config: TabularPipelineConfig) -> Pipeline:
    """Build the numeric sub-pipeline (imputer + optional scaler)."""
    numeric_steps: List[tuple[str, object]] = []

    imputer_kwargs: dict = {"strategy": config.numeric_imputer_strategy}
    if config.numeric_imputer_strategy == "constant":
        imputer_kwargs["fill_value"] = (
            config.numeric_imputer_fill_value if config.numeric_imputer_fill_value is not None else 0.0
        )

    numeric_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))

    if config.scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return Pipeline(steps=numeric_steps)


def _build_categorical_pipeline(config: TabularPipelineConfig) -> Pipeline:
    """Build the categorical sub-pipeline (imputer + OneHotEncoder)."""
    cat_steps: List[tuple[str, object]] = []

    imputer_kwargs: dict = {"strategy": config.categorical_imputer_strategy}
    if config.categorical_imputer_strategy == "constant":
        imputer_kwargs["fill_value"] = (
            config.categorical_imputer_fill_value
            if config.categorical_imputer_fill_value is not None
            else "missing"
        )

    cat_steps.append(("imputer", SimpleImputer(**imputer_kwargs)))

    # sklearn >= 1.4 supports sparse_output and min_frequency / max_categories
    encoder_kwargs: dict = {
        "handle_unknown": config.ohe_handle_unknown,
        "sparse_output": False,
    }
    if config.ohe_min_frequency is not None:
        encoder_kwargs["min_frequency"] = config.ohe_min_frequency
    if config.ohe_max_categories is not None:
        encoder_kwargs["max_categories"] = config.ohe_max_categories

    cat_steps.append(("onehot", OneHotEncoder(**encoder_kwargs)))

    return Pipeline(steps=cat_steps)


def build_tabular_preprocessor(
    df_sample: pd.DataFrame,
    config: Optional[TabularPipelineConfig] = None,
) -> ColumnTransformer:
    """Build a ColumnTransformer-based preprocessing pipeline for tabular data.

    Parameters
    ----------
    df_sample:
        A representative DataFrame (e.g., your training data) used to infer
        default numeric/categorical columns and validate configured lists.
    config:
        Optional TabularPipelineConfig. If None, a default configuration is used.

    Returns
    -------
    ColumnTransformer
        A scikit-learn ColumnTransformer that can be fit/transform-ed on DataFrames
        with the same schema as df_sample.

    Notes
    -----
    - The returned transformer operates on pandas DataFrames and outputs a numpy array.
    - For integration with PyTorch, you typically call:

          preprocessor = build_tabular_preprocessor(df_train, cfg)
          X_train = preprocessor.fit_transform(df_train)
          X_val = preprocessor.transform(df_val)

      and then feed X_train / X_val into your torch datasets.
    """
    if config is None:
        config = TabularPipelineConfig()

    numeric_features, categorical_features = _infer_feature_types(df_sample, config)

    transformers: List[tuple[str, object, Iterable[str]]] = []

    if numeric_features:
        numeric_pipeline = _build_numeric_pipeline(config)
        transformers.append(("numeric", numeric_pipeline, numeric_features))

    if categorical_features:
        categorical_pipeline = _build_categorical_pipeline(config)
        transformers.append(("categorical", categorical_pipeline, categorical_features))

    # If no explicit numeric/categorical features but remainder != "drop",
    # ColumnTransformer will pass through any remaining columns unchanged.
    if not transformers and config.remainder == "drop":
        # This is a safety net; _infer_feature_types would have raised already in this case.
        raise DataError(
            "No transformers configured and remainder='drop'; nothing to transform.",
            code="no_transformers_configured",
            context={},
            location="ml_tabular.sklearn_utils.tabular_pipeline.build_tabular_preprocessor",
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=config.remainder,
        verbose=False,
    )

    logger.info(
        "Built tabular preprocessor with %d transformer(s), remainder=%s",
        len(transformers),
        config.remainder,
    )

    return preprocessor


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def fit_preprocessor(
    X_train: pd.DataFrame,
    config: Optional[TabularPipelineConfig] = None,
) -> ColumnTransformer:
    """Build and fit a tabular preprocessor on the training DataFrame.

    This is a convenience wrapper around build_tabular_preprocessor + fit().

    Parameters
    ----------
    X_train:
        Training DataFrame.
    config:
        Optional TabularPipelineConfig.

    Returns
    -------
    ColumnTransformer
        A fitted ColumnTransformer.
    """
    preprocessor = build_tabular_preprocessor(X_train, config=config)
    logger.info("Fitting tabular preprocessor on training data with shape: %s", X_train.shape)
    preprocessor.fit(X_train)
    return preprocessor


def fit_transform_preprocessor(
    X_train: pd.DataFrame,
    config: Optional[TabularPipelineConfig] = None,
) -> Tuple[np.ndarray, ColumnTransformer]:
    """Build, fit, and transform training data with the preprocessor.

    Parameters
    ----------
    X_train:
        Training DataFrame.
    config:
        Optional TabularPipelineConfig.

    Returns
    -------
    X_transformed:
        Numpy array of transformed training features.
    preprocessor:
        Fitted ColumnTransformer that can be reused for validation/test data.
    """
    preprocessor = build_tabular_preprocessor(X_train, config=config)
    logger.info("Fitting + transforming training data with preprocessor.")
    X_transformed = preprocessor.fit_transform(X_train)
    return X_transformed, preprocessor


def transform_with_preprocessor(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Transform a DataFrame using an already-fitted preprocessor.

    Parameters
    ----------
    preprocessor:
        A fitted ColumnTransformer returned by fit_preprocessor or
        fit_transform_preprocessor.
    X:
        DataFrame to transform. Must have the same schema as the training DataFrame
        used to fit the preprocessor (at least for the columns the preprocessor expects).

    Returns
    -------
    np.ndarray
        Transformed feature matrix.
    """
    # Basic sanity check: ensure expected columns are present
    expected_cols: Optional[Sequence[str]] = None
    if isinstance(preprocessor.transformers, list):
        # Collect all explicitly configured columns across transformers
        cols: List[str] = []
        for _, _, cols_spec in preprocessor.transformers:
            # cols_spec may be a list of names or a mask; we only handle the list case here
            if isinstance(cols_spec, (list, tuple)):
                cols.extend(cols_spec)
        expected_cols = cols or None

    if expected_cols is not None:
        missing = [c for c in expected_cols if c not in X.columns]
        if missing:
            raise DataError(
                "Input DataFrame is missing columns expected by the preprocessor.",
                code="missing_columns_for_preprocessor",
                context={"missing": missing, "columns": list(X.columns)},
                location="ml_tabular.sklearn_utils.tabular_pipeline.transform_with_preprocessor",
            )

    logger.info("Transforming data with preprocessor; input shape: %s", X.shape)
    X_transformed = preprocessor.transform(X)
    return np.asarray(X_transformed)
