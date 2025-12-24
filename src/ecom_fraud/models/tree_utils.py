# ecom_fraud/models/tree_utils.py

from __future__ import annotations
from typing import Sequence, Tuple, Dict, Optional

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from ecom_fraud.mlops.mlflow_utils import (
    mlflow_run,
    log_params,
    log_metrics,
)


def pr_auc_scorer(estimator, X, y_true):
    proba = estimator.predict_proba(X)[:, 1]
    return average_precision_score(y_true, proba)


def build_tabular_preprocessor(
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
) -> ColumnTransformer:
    """
    Build a reusable ColumnTransformer for tabular data:
      - one-hot encodes categorical columns
      - passes numeric columns through unchanged
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)),
            ("num", "passthrough", list(numeric_cols)),
        ]
    )
    return preprocessor


def build_model_pipeline(
    base_estimator,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
    preprocessor: Optional[ColumnTransformer] = None,
) -> Tuple[Pipeline, ColumnTransformer]:
    """
    Wrap any sklearn-style estimator in a preprocessing + model Pipeline.

    Parameters
    ----------
    base_estimator :
        Unfitted estimator with .fit/.predict_proba (e.g. LGBMClassifier, XGBClassifier).
    categorical_cols :
        Column names to be one-hot encoded.
    numeric_cols :
        Column names to be passed through as numeric features.
    preprocessor :
        Optionally pass an existing ColumnTransformer to reuse across models.
        If None, a new one is created from categorical_cols / numeric_cols.

    Returns
    -------
    pipeline :
        sklearn Pipeline: [preprocess -> model].
    preprocessor :
        The ColumnTransformer used inside the pipeline (for reuse elsewhere).
    """
    if preprocessor is None:
        preprocessor = build_tabular_preprocessor(categorical_cols, numeric_cols)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_estimator),
        ]
    )
    return pipeline, preprocessor


def build_time_series_oof_scores(
    base_estimator,
    preprocessor,
    X_train_base: pd.DataFrame,
    y_train: pd.Series,
    X_val_base: pd.DataFrame,
    y_val: pd.Series,
    n_splits: int = 5,
    tree_name: str = "tree",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.Series]:
    """
    Build history-safe OOF scores on train+val using TimeSeriesSplit.

    Parameters
    ----------
    base_estimator :
        Unfitted sklearn-style estimator (e.g. LGBMClassifier, XGBClassifier).
    preprocessor :
        ColumnTransformer (will be cloned inside each fold).
    X_train_base, y_train :
        Training features and labels (time-sorted).
    X_val_base, y_val :
        Validation features and labels (time-sorted).
    n_splits :
        Number of TimeSeriesSplit folds.
    tree_name :
        Label used in logs.

    Returns
    -------
    oof_preds_train : np.ndarray
        OOF probabilities for the train portion (may have burn-in NaNs removed later).
    val_tree_score : np.ndarray
        OOF probabilities for the val portion (no NaNs expected).
    X_trainval_base : pd.DataFrame
        Concatenation of X_train_base and X_val_base (time-ordered).
    y_trainval : pd.Series
        Concatenation of y_train and y_val (time-ordered).
    """
    # Combine train + val for OOF generation
    X_trainval_base = pd.concat([X_train_base, X_val_base], axis=0).reset_index(drop=True)
    y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    n_train = len(X_train_base)
    n_val = len(X_val_base)

    tscv_oof = TimeSeriesSplit(n_splits=n_splits)

    # Start with NaNs so "never predicted" rows are explicit (burn-in window)
    oof_preds_trainval = np.full(len(X_trainval_base), np.nan, dtype=np.float32)

    print(f"[{tree_name}] Building OOF scores with TimeSeriesSplit(n_splits={n_splits})")

    for fold, (tr_idx, val_idx) in enumerate(tscv_oof.split(X_trainval_base), start=1):
        print(
            f"[{tree_name}] OOF fold {fold}/{tscv_oof.n_splits}: "
            f"train={len(tr_idx)} val={len(val_idx)}"
        )

        pipe = Pipeline(
            steps=[
                ("preprocess", clone(preprocessor)),   # fresh preprocessor each fold
                ("model", clone(base_estimator)),      # fresh model each fold
            ]
        )

        X_tr_fold = X_trainval_base.iloc[tr_idx]
        y_tr_fold = y_trainval.iloc[tr_idx]
        X_val_fold = X_trainval_base.iloc[val_idx]

        pipe.fit(X_tr_fold, y_tr_fold)
        proba_val_fold = pipe.predict_proba(X_val_fold)[:, 1].astype(np.float32)

        # Fill OOF positions for this fold
        oof_preds_trainval[val_idx] = proba_val_fold

    print(f"[{tree_name}] OOF preds (train+val) shape:", oof_preds_trainval.shape)

    # Score only on rows that actually received an OOF prediction (exclude NaNs)
    pred_mask = ~np.isnan(oof_preds_trainval)
    print(f"[{tree_name}] Never-predicted rows (burn-in):", (~pred_mask).sum())
    print(f"[{tree_name}] First missing indices:", np.where(~pred_mask)[0][:10])

    print(
        f"[{tree_name}] OOF train+val PR-AUC (internal; excluding burn-in):",
        average_precision_score(y_trainval[pred_mask], oof_preds_trainval[pred_mask]),
    )

    # Slice back into train and val parts
    oof_preds_train = oof_preds_trainval[:n_train]
    val_tree_score = oof_preds_trainval[n_train:]

    # Expect val to be fully covered; if not, something is off with sizes/splits
    assert not np.isnan(val_tree_score).any(), (
        f"[{tree_name}] Val has NaN tree_scores; check TimeSeriesSplit coverage."
    )

    return oof_preds_train, val_tree_score, X_trainval_base, y_trainval

def build_stacked_tree_features(
    df_train_ml: pd.DataFrame,
    df_val_ml: pd.DataFrame,
    df_test_ml: pd.DataFrame,
    y_col: str,
    oof_scores_dict: Dict[str, np.ndarray],
    val_scores_dict: Dict[str, np.ndarray],
    test_scores_dict: Dict[str, np.ndarray],
    drop_burn_in: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Add one or more tree-based score columns to train/val/test.

    oof_scores_dict / val_scores_dict / test_scores_dict are mappings
    like {"lgbm": scores, "xgb": scores}. Keys become column suffixes.
    """

    # assume all OOF arrays same length
    n = len(next(iter(oof_scores_dict.values())))
    burn_mask = np.ones(n, dtype=bool)

    for scores in oof_scores_dict.values():
        burn_mask &= ~np.isnan(scores)

    if drop_burn_in:
        n_dropped = (~burn_mask).sum()
        print(f"Dropping {n_dropped} burn-in train rows with no OOF scores.")
    else:
        burn_mask[:] = True

    df_train_stack = df_train_ml.iloc[burn_mask].copy().reset_index(drop=True)
    df_val_stack = df_val_ml.copy()
    df_test_stack = df_test_ml.copy()

    for name, scores in oof_scores_dict.items():
        col = f"{name}_score"
        df_train_stack[col] = scores[burn_mask].astype(np.float32)

    for name, scores in val_scores_dict.items():
        col = f"{name}_score"
        df_val_stack[col] = scores.astype(np.float32)

    for name, scores in test_scores_dict.items():
        col = f"{name}_score"
        df_test_stack[col] = scores.astype(np.float32)

    return df_train_stack, df_val_stack, df_test_stack


def fit_tree_trainval_and_score_test(
    base_estimator,
    preprocessor,
    X_trainval_base: pd.DataFrame,
    y_trainval: pd.Series,
    X_test_base: pd.DataFrame,
    y_test: pd.Series,
    tree_name: str = "tree",
):
    """
    Fit tree model on (train+val) once and evaluate on test.

    Parameters
    ----------
    base_estimator :
        Unfitted estimator (same type / hyperparams used for OOF).
    preprocessor :
        ColumnTransformer (cloned before fitting).
    X_trainval_base, y_trainval :
        Features and labels for train+val.
    X_test_base, y_test :
        Test features and labels.
    tree_name :
        Label used in logs.

    Returns
    -------
    test_tree_score : np.ndarray
        Predicted probabilities P(y=1) on the test set.
    test_pr_auc_tree : float
        PR-AUC on the test set.
    test_roc_auc_tree : float
        ROC-AUC on the test set.
    """
    tree_pipeline = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),    # fit preprocess only on train+val
            ("model", clone(base_estimator)),
        ]
    )

    print(f"[{tree_name}] Fitting on train+val and scoring test...")
    tree_pipeline.fit(X_trainval_base, y_trainval)

    test_tree_score = tree_pipeline.predict_proba(X_test_base)[:, 1].astype(np.float32)
    print(f"[{tree_name}] Test tree_score shape:", test_tree_score.shape)

    # Tree-only baseline on test for comparison
    y_test_proba_tree = test_tree_score
    y_test_pred_tree = (y_test_proba_tree >= 0.5).astype("int8")

    test_pr_auc_tree = average_precision_score(y_test, y_test_proba_tree)
    test_roc_auc_tree = roc_auc_score(y_test, y_test_proba_tree)

    print(f"[{tree_name}] Test PR-AUC:  {test_pr_auc_tree:.4f}")
    print(f"[{tree_name}] Test ROC-AUC: {test_roc_auc_tree:.4f}")
    print(f"{tree_name} test classification report (thr=0.5):")
    print(classification_report(y_test, y_test_pred_tree, digits=4))

    return test_tree_score, test_pr_auc_tree, test_roc_auc_tree


def run_tree_random_search_with_mlflow(
    cfg,
    X_train_base: pd.DataFrame,
    y_train: pd.Series,
    base_pipeline: Pipeline,
    param_distributions: dict,
    tscv: TimeSeriesSplit,
    model_family: str,
    run_name: str,
    n_iter: int = 20,
):
    """
    Run RandomizedSearchCV for a tree model with PR-AUC scoring and MLflow logging.
    Returns (best_pipeline, best_params, best_score).
    """
    with mlflow_run(
        cfg,
        experiment_name="gbdt_hybrid",
        run_name=run_name,
        tags={
            "model_family": model_family,
            "problem_type": "fraud_detection",
            "dataset": "kaggle_ecom_fraud",
            "stage": "hyperparam_search_tree",
        },
    ) as mlflow_ctx:

        search = RandomizedSearchCV(
            estimator=base_pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            scoring=pr_auc_scorer,
            cv=tscv,
            n_jobs=-1,
            verbose=2,
            random_state=getattr(getattr(cfg, "training", None), "random_seed", 42),
            refit=True,
        )

        search.fit(X_train_base, y_train)

        print(f"[{model_family}] Best PR-AUC (CV):", search.best_score_)
        print(f"[{model_family}] Best params:")
        for k, v in search.best_params_.items():
            print(f"  {k}: {v}")

        if mlflow_ctx is not None:
            log_params(search.best_params_, prefix=f"{model_family.lower()}_tuned.")
            log_metrics({"cv_best_pr_auc": search.best_score_})

    return search.best_estimator_, search.best_params_, search.best_score_


