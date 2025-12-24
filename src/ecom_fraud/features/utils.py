# ecom_fraud/features/utils.py

from __future__ import annotations
import pandas as pd


def infer_feature_types(X: pd.DataFrame):
    """
    Infer categorical vs numeric feature columns for the tabular models.
    """
    categorical_cols = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_cols = X.select_dtypes(
        include=[
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "Int64",
        ]
    ).columns.tolist()

    print("Categorical columns:", categorical_cols)
    print("Numeric columns:", len(numeric_cols))

    return categorical_cols, numeric_cols
