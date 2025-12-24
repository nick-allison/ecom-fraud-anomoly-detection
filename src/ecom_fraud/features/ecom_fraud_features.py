# ecom_fraud/features/ecom_fraud_features.py

from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

from ecom_fraud.features.tabular import FeatureSpec, build_features
from ecom_fraud.features.time_series import (
    TimeSeriesFeatureSpec,
    build_time_series_features,
)


def add_handcrafted_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple flags based on earlier EDA:
    - country_mismatch
    - is_web / is_app
    - is_low_amount / is_high_amount
    - is_high_distance
    - no_3ds
    """
    df = df.copy()

    # paste the Cell 1 block here, but operate on df, not df_fe:
    # 1. Country mismatch between card BIN and shipping country
    df["country_mismatch"] = (df["country"] != df["bin_country"]).astype("int8")
    # 2. Channel flags
    df["is_web"] = (df["channel"] == "web").astype("int8")
    df["is_app"] = (df["channel"] == "app").astype("int8")
    # 3. High-risk amount flags
    df["is_low_amount"] = (df["amount"] <= 21.03).astype("int8")
    df["is_high_amount"] = (df["amount"] >= 384.626).astype("int8")
    # 4. High-risk distance flags
    df["is_high_distance"] = (df["shipping_distance_km"] >= 491.04).astype("int8")
    # 5. No 3DS flag
    df["no_3ds"] = (df["three_ds_flag"] == 0).astype("int8")

    return df


def build_tabular_features(
    df: pd.DataFrame,
    dataset_name: str = "full_transactions",
) -> pd.DataFrame:
    """
    Apply the FeatureSpec used in the notebook to build tabular features.
    Mirrors the existing Cell 2 code.
    """
    tab_spec = FeatureSpec(
        datetime_columns=["transaction_time"],
        log1p_columns=["amount", "avg_amount_user", "shipping_distance_km"],
        ratio_features={
            "amount_to_avg_ratio": ("amount", "avg_amount_user"),
            "distance_per_amount": ("shipping_distance_km", "amount"),
        },
        power_features={
            "amount_sqrt": ("amount", 0.5),
            "distance_sqrt": ("shipping_distance_km", 0.5),
        },
        interaction_features={
            "web_x_mismatch": ("is_web", "country_mismatch"),
            "web_x_no_3ds": ("is_web", "no_3ds"),
            "web_x_promo": ("is_web", "promo_used"),
            "high_amt_x_high_dist": ("is_high_amount", "is_high_distance"),
        },
        drop_original_datetime=False,
    )

    df_tab = build_features(df, dataset_name=dataset_name, spec=tab_spec)
    return df_tab


def build_time_series_features_per_user(
    df_tab: pd.DataFrame,
    dataset_name: str = "full_transactions",
) -> pd.DataFrame:
    """
    Apply the TimeSeriesFeatureSpec used in the notebook and drop NA lags.
    Mirrors Cell 3.
    """
    ts_spec = TimeSeriesFeatureSpec(
        datetime_column="transaction_time",
        group_column="user_id",
        sort_by_time=True,
        expand_datetime=False,
        drop_original_datetime=False,
        lag_features={
            "amount_lag1": ("amount", 1),
            "amount_lag3": ("amount", 3),
            "user_fraud_lag1": ("is_fraud", 1),
        },
        rolling_features={
            "user_amount_roll_mean_5":  ("amount", 5,  "mean"),
            "user_amount_roll_std_5":   ("amount", 5,  "std"),
            "user_amount_roll_sum_5":   ("amount", 5,  "sum"),
            "user_amount_roll_mean_10": ("amount", 10, "mean"),
            "user_amount_roll_std_10":  ("amount", 10, "std"),
            "user_amount_roll_sum_10":  ("amount", 10, "sum"),
            "user_amount_roll_mean_30": ("amount", 30, "mean"),
            "user_amount_roll_std_30":  ("amount", 30, "std"),
            "user_amount_roll_sum_30":  ("amount", 30, "sum"),
            "user_txn_roll_sum_3":  ("is_fraud", 3,  "sum"),
            "user_txn_roll_sum_5":  ("is_fraud", 5,  "sum"),
            "user_txn_roll_sum_10": ("is_fraud", 10, "sum"),
            "user_txn_roll_sum_30": ("is_fraud", 30, "sum"),
        },
        expanding_features={},
        rolling_min_periods=1,
        history_safe=True,
    )

    df_ts = build_time_series_features(
        df_tab,
        dataset_name=dataset_name,
        spec=ts_spec,
    )

    lag_cols = list(ts_spec.lag_features.keys())
    df_model = df_ts.dropna(subset=lag_cols).reset_index(drop=True)
    return df_model


def add_history_safe_user_geo_security_features(
    df_model: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add history-safe user / geo / security behaviour features.
    This is exactly the code from Cell 4, just wrapped into a function.
    """
    df_feat = df_model.copy()
    df_feat = df_feat.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)
    g_user = df_feat.groupby("user_id", group_keys=False)

    # -- paste everything from Cell 4 here, but use df_feat consistently --
    # (secs_since_last_tx, secs_since_last_fraud, sec_bad, user_sec_bad_rate_10/30,
    #  secs_since_last_sec_issue, distance behaviours, amount behaviours,
    #  geo patterns, time-of-day patterns, extra interactions, etc.)

    df_model_out = df_feat.sort_values(["user_id", "transaction_time"]).reset_index(drop=True)
    return df_model_out


def drop_optional_stage1_columns(df_model: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns listed in drop_cols_stage1 (Cell 5).
    """
    drop_cols_stage1 = [
        "transaction_time__year",
        "transaction_time__is_weekend",
        "transaction_time__is_month_start",
        "transaction_time__is_month_end",
        "is_low_amount",
        "is_high_amount",
        "is_new_country_for_user",
        "country_changed_from_last",
        "new_country_x_high_amount",
        "is_unusual_hour_for_user",
        "high_amount_x_no_3ds",
        "app_x_high_amount",
        "user_fraud_lag1",
    ]
    cols_to_drop_extra = [c for c in drop_cols_stage1 if c in df_model.columns]
    return df_model.drop(columns=cols_to_drop_extra)


def build_ecom_fraud_feature_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature pipeline used in the notebook:
    raw df -> handcrafted flags -> tabular features -> TS features ->
    history-safe user/geo/security features -> optional drops.
    """
    df = df_raw.copy()

    # Ensure transaction_time is datetime
    df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")

    df = add_handcrafted_flags(df)
    df_tab = build_tabular_features(df)
    df_model = build_time_series_features_per_user(df_tab)
    df_model = add_history_safe_user_geo_security_features(df_model)
    df_model = drop_optional_stage1_columns(df_model)

    return df_model
