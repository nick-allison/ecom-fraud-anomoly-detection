# ecom_fraud/torch/tabular_inputs.py

from __future__ import annotations
from typing import Sequence, Dict, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def prepare_embedding_mlp_inputs(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    embedding_cats: Sequence[str],
    scaler: StandardScaler | None = None,
) -> Tuple[
    np.ndarray,  # X_train_num
    np.ndarray,  # X_val_num
    np.ndarray,  # X_test_num
    np.ndarray,  # X_train_cat
    np.ndarray,  # X_val_cat
    np.ndarray,  # X_test_cat
    list[str],   # numeric_cols_for_mlp
    list[str],   # embedding_cats_used
    list[int],   # cat_cardinalities
    list[int],   # embedding_dims
    StandardScaler,          # fitted scaler
    Dict[str, list[str]],    # cat_vocab
]:
    """
    Prepare numeric + categorical arrays for the embedding MLP.

    Assumes X_train / X_val / X_test are *feature-only* (TARGET_COL already dropped).

    Parameters
    ----------
    X_train, X_val, X_test :
        Feature matrices (pandas DataFrames).
    embedding_cats :
        List of column names to treat as categorical for embeddings.
        Columns not present in X_train are silently ignored.
    scaler :
        Optional existing StandardScaler. If None, a new one is fit on X_train.

    Returns
    -------
    X_train_num, X_val_num, X_test_num :
        float32 numeric feature matrices after standardization.
    X_train_cat, X_val_cat, X_test_cat :
        int64 categorical index matrices (for embeddings).
    numeric_cols_for_mlp :
        Names of the numeric columns used (including tree_score).
    embedding_cats_used :
        Final list of categorical columns actually used.
    cat_cardinalities :
        Number of classes per categorical feature (for embedding sizes).
    embedding_dims :
        Embedding dimension per categorical feature.
    scaler :
        Fitted StandardScaler instance.
    cat_vocab :
        Mapping col_name -> list of category labels (for future inference).
    """
    # Copies we can mutate
    X_train_df = X_train.copy()
    X_val_df = X_val.copy()
    X_test_df = X_test.copy()

    # Filter embedding_cats to those actually present
    embedding_cats_used = [c for c in embedding_cats if c in X_train_df.columns]

    cat_vocab: Dict[str, list[str]] = {}
    cat_num_classes: Dict[str, int] = {}

    # 1) Build category vocab + *_idx columns with shared vocab across splits
    for col in embedding_cats_used:
        # Train: fit categories
        X_train_df[col] = X_train_df[col].astype("category")
        cat_vocab[col] = list(X_train_df[col].cat.categories)
        X_train_df[col + "_idx"] = X_train_df[col].cat.codes

        # Val/Test: align to train vocab
        X_val_df[col] = pd.Categorical(X_val_df[col], categories=cat_vocab[col])
        X_val_df[col + "_idx"] = X_val_df[col].cat.codes

        X_test_df[col] = pd.Categorical(X_test_df[col], categories=cat_vocab[col])
        X_test_df[col + "_idx"] = X_test_df[col].cat.codes

    # 2) Reserve explicit "unknown" index (for any -1 codes)
    for col in embedding_cats_used:
        train_codes = X_train_df[col + "_idx"].to_numpy()
        val_codes = X_val_df[col + "_idx"].to_numpy()
        test_codes = X_test_df[col + "_idx"].to_numpy()

        max_code = max(
            train_codes[train_codes >= 0].max(initial=-1),
            val_codes[val_codes >= 0].max(initial=-1),
            test_codes[test_codes >= 0].max(initial=-1),
        )
        unk_index = max_code + 1

        train_codes[train_codes < 0] = unk_index
        val_codes[val_codes < 0] = unk_index
        test_codes[test_codes < 0] = unk_index

        X_train_df[col + "_idx"] = train_codes
        X_val_df[col + "_idx"] = val_codes
        X_test_df[col + "_idx"] = test_codes

        cat_num_classes[col] = int(unk_index + 1)

    print("Categorical embedding vocab sizes:", cat_num_classes)

    # 3) Numeric columns (include tree_score; exclude original cat + *_idx)
    numeric_cols_for_mlp = [
        c
        for c in X_train_df.columns
        if c not in embedding_cats_used and not c.endswith("_idx")
    ]

    if scaler is None:
        scaler = StandardScaler()

    X_train_num = scaler.fit_transform(X_train_df[numeric_cols_for_mlp].to_numpy())
    X_val_num = scaler.transform(X_val_df[numeric_cols_for_mlp].to_numpy())
    X_test_num = scaler.transform(X_test_df[numeric_cols_for_mlp].to_numpy())

    cat_idx_cols = [c + "_idx" for c in embedding_cats_used]

    X_train_cat = X_train_df[cat_idx_cols].to_numpy().astype("int64")
    X_val_cat = X_val_df[cat_idx_cols].to_numpy().astype("int64")
    X_test_cat = X_test_df[cat_idx_cols].to_numpy().astype("int64")

    X_train_num = X_train_num.astype("float32")
    X_val_num = X_val_num.astype("float32")
    X_test_num = X_test_num.astype("float32")

    print(
        "Numeric shapes: train / val / test =",
        X_train_num.shape,
        X_val_num.shape,
        X_test_num.shape,
    )
    print(
        "Cat idx shapes: train / val / test =",
        X_train_cat.shape,
        X_val_cat.shape,
        X_test_cat.shape,
    )

    # 4) Embedding cardinalities & dims
    cat_cardinalities: list[int] = []
    for col in embedding_cats_used:
        n_unique = int(X_train_df[col + "_idx"].max()) + 1
        cat_cardinalities.append(n_unique)

    def emb_dim(n_cat: int) -> int:
        return min(32, max(4, int(round(np.sqrt(n_cat)))))

    embedding_dims = [emb_dim(n) for n in cat_cardinalities]

    print("cat_cardinalities:", cat_cardinalities)
    print("embedding_dims:", embedding_dims)
    print("Numeric features (including tree_score):", len(numeric_cols_for_mlp))

    return (
        X_train_num,
        X_val_num,
        X_test_num,
        X_train_cat,
        X_val_cat,
        X_test_cat,
        numeric_cols_for_mlp,
        embedding_cats_used,
        cat_cardinalities,
        embedding_dims,
        scaler,
        cat_vocab,
    )
