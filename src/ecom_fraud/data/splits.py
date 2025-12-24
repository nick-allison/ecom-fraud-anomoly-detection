# ecom_fraud/data/splits.py

from __future__ import annotations
from typing import List, Tuple
import pandas as pd


def time_based_train_val_test_split(
    df: pd.DataFrame,
    time_col: str,
    test_frac: float = 0.2,
    val_frac: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Sort by `time_col` and perform contiguous time-based splits:
      - earliest -> train
      - middle   -> val
      - latest   -> test

    Fractions are relative to the full dataset:
      - last `test_frac` goes to test
      - from the remaining, last `val_frac` goes to val
    """
    df_sorted = df.copy()
    df_sorted[time_col] = pd.to_datetime(df_sorted[time_col], errors="raise")
    df_sorted = df_sorted.sort_values(time_col).reset_index(drop=True)

    n_rows = len(df_sorted)
    test_start_idx = int(n_rows * (1 - test_frac))

    df_trainval = df_sorted.iloc[:test_start_idx].reset_index(drop=True)
    df_test = df_sorted.iloc[test_start_idx:].reset_index(drop=True)

    n_trainval = len(df_trainval)
    val_start_idx = int(n_trainval * (1 - val_frac))

    df_train = df_trainval.iloc[:val_start_idx].reset_index(drop=True)
    df_val = df_trainval.iloc[val_start_idx:].reset_index(drop=True)

    print("Train rows:", df_train.shape[0])
    print("Val rows:  ", df_val.shape[0])
    print("Test rows: ", df_test.shape[0])

    print(
        "Train time range:",
        df_train[time_col].min(),
        "->",
        df_train[time_col].max(),
    )
    print(
        "Val time range:  ",
        df_val[time_col].min(),
        "->",
        df_val[time_col].max(),
    )
    print(
        "Test time range: ",
        df_test[time_col].min(),
        "->",
        df_test[time_col].max(),
    )

    # Guard against any accidental leakage
    assert df_train[time_col].max() < df_val[time_col].min()
    assert df_val[time_col].max() < df_test[time_col].min()

    return df_train, df_val, df_test


def build_ml_matrices(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    target_col: str,
    id_cols: List[str],
    raw_time_cols: List[str],
):
    """
    Drop ID / raw time columns and split into:
      - df_*_ml (model-ready DFs)
      - X_*_base, y_*
    """
    cols_to_drop = [c for c in (id_cols + raw_time_cols) if c in df_train.columns]

    df_train_ml = df_train.drop(columns=cols_to_drop)
    df_val_ml = df_val.drop(columns=cols_to_drop)
    df_test_ml = df_test.drop(columns=cols_to_drop)

    X_train_base = df_train_ml.drop(columns=[target_col])
    y_train = df_train_ml[target_col].astype("int8")

    X_val_base = df_val_ml.drop(columns=[target_col])
    y_val = df_val_ml[target_col].astype("int8")

    X_test_base = df_test_ml.drop(columns=[target_col])
    y_test = df_test_ml[target_col].astype("int8")

    print("Train fraud ratio:", y_train.mean())
    print("Val fraud ratio:  ", y_val.mean())
    print("Test fraud ratio: ", y_test.mean())

    return (
        df_train_ml,
        df_val_ml,
        df_test_ml,
        X_train_base,
        X_val_base,
        X_test_base,
        y_train,
        y_val,
        y_test,
    )
