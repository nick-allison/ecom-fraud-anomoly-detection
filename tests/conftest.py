from __future__ import annotations

import random
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


# ---------------------------------------------------------------------------
# Global test seed
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_test_seed() -> Generator[None, None, None]:
    """Set a deterministic random seed for every test.

    This keeps small numeric tests (loss going down, etc.) more stable. If a
    test needs its own custom seed, it can override inside the test.
    """
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    yield


# ---------------------------------------------------------------------------
# In-memory DataFrames
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_tabular_df() -> pd.DataFrame:
    """Small tabular dataset: 3 rows, 2 numeric features, 1 numeric target.

    This is intentionally tiny and generic so it can be reused across tests
    (datasets, models, training loops, etc.).
    """
    return pd.DataFrame(
        {
            "feat1": [1.0, 2.0, 3.0],
            "feat2": [0.5, 1.5, 2.5],
            "target": [0.0, 1.0, 0.0],
        }
    )


@pytest.fixture
def simple_time_series_df() -> pd.DataFrame:
    """Small time-series dataset with a datetime column, 2 features, 1 target.

    Used to test TimeSeriesSequenceDataset, windowing logic, and time-series
    configs without bringing in a real-world dataset.
    """
    dates = pd.date_range("2020-01-01", periods=12, freq="H")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "feat1": np.arange(len(dates), dtype=float),
            "feat2": np.arange(len(dates), dtype=float) * 2.0,
            "target": (np.arange(len(dates)) % 3).astype(float),
        }
    )


# ---------------------------------------------------------------------------
# On-disk datasets under a temporary base dir
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_tabular_dataset(tmp_path: Path, simple_tabular_df: pd.DataFrame) -> Path:
    """Create a tiny tabular CSV under a temporary data/ directory.

    Returns the full path to the CSV. This is useful for tests that want to
    exercise the *filesystem* side of data loading (e.g. using PathsConfig).
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "tabular_example.csv"
    simple_tabular_df.to_csv(csv_path, index=False)
    return csv_path


@pytest.fixture
def tmp_time_series_dataset(tmp_path: Path, simple_time_series_df: pd.DataFrame) -> Path:
    """Create a tiny time-series CSV under a temporary data/ directory.

    Returns the full path to the CSV.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    csv_path = data_dir / "time_series_example.csv"
    # Ensure the timestamp column is written in ISO format
    df = simple_time_series_df.copy()
    df["timestamp"] = df["timestamp"].astype(str)
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------------------------------------------------------------------
# Temporary YAML configs that match the library's expectations
# ---------------------------------------------------------------------------


@pytest.fixture
def tabular_config_path(tmp_path: Path, tmp_tabular_dataset: Path) -> Path:
    """Write a minimal-but-realistic tabular config YAML under tmp_path.

    The config is shaped to align with ml_tabular.config.AppConfig and the
    tabular training script:

      env: "dev"
      experiment_name: "test_tabular"
      paths:
        base_dir: "<tmp_path>"
        data_dir: "data"
        raw_dir: "data/raw"
        processed_dir: "data/processed"
        models_dir: "models"
      training:
        random_seed: 42
        batch_size: 16
        num_epochs: 3
        learning_rate: 0.001
        weight_decay: 0.0001
        optimizer: "adamw"
        hidden_dims: [32, 16]
      tabular:
        dataset_csv: "tabular_example.csv"
        target_column: "target"
        feature_columns: ["feat1", "feat2"]
        test_size: 0.2
        task_type: "binary"

    Returns the path to the YAML file.
    """
    base_dir = tmp_path
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    # Ensure directories exist (they may be used by code under test)
    for d in (data_dir, models_dir, raw_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = {
        "env": "dev",
        "experiment_name": "test_tabular",
        "log_level": "INFO",
        "paths": {
            "base_dir": str(base_dir),
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "models",
        },
        "training": {
            "random_seed": 42,
            "batch_size": 16,
            "num_epochs": 3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "hidden_dims": [32, 16],
            "activation": "relu",
            "dropout": 0.0,
            "batch_norm": False,
            "layer_norm": False,
            # simple defaults for other knobs if your AppConfig expects them
            "early_stopping_enabled": False,
        },
        "tabular": {
            "dataset_csv": tmp_tabular_dataset.name,  # "tabular_example.csv"
            "target_column": "target",
            "feature_columns": ["feat1", "feat2"],
            "test_size": 0.2,
            "task_type": "binary",
        },
    }

    config_path = tmp_path / "tabular_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


@pytest.fixture
def time_series_config_path(tmp_path: Path, tmp_time_series_dataset: Path) -> Path:
    """Write a minimal time-series config YAML under tmp_path.

    The config is shaped to align with the time-series training code:

      env: "dev"
      experiment_name: "test_ts"
      paths:
        base_dir: "<tmp_path>"
        data_dir: "data"
        ...
      training:
        random_seed: 42
        batch_size: 8
        num_epochs: 3
        learning_rate: 0.001
        weight_decay: 0.0001
        optimizer: "adamw"
        hidden_dims: [64, 32]
      time_series:
        dataset_csv: "time_series_example.csv"
        datetime_column: "timestamp"
        target_column: "target"
        value_columns: ["feat1", "feat2"]
        val_fraction: 0.2
        input_window: 3
        prediction_horizon: 1
        step_size: 1
        task_type: "regression"

    Returns the path to the YAML file.
    """
    base_dir = tmp_path
    data_dir = base_dir / "data"
    models_dir = base_dir / "models"
    raw_dir = base_dir / "data" / "raw"
    processed_dir = base_dir / "data" / "processed"

    for d in (data_dir, models_dir, raw_dir, processed_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = {
        "env": "dev",
        "experiment_name": "test_ts",
        "log_level": "INFO",
        "paths": {
            "base_dir": str(base_dir),
            "data_dir": "data",
            "raw_dir": "data/raw",
            "processed_dir": "data/processed",
            "models_dir": "models",
        },
        "training": {
            "random_seed": 42,
            "batch_size": 8,
            "num_epochs": 3,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "optimizer": "adamw",
            "hidden_dims": [64, 32],
            "activation": "relu",
            "dropout": 0.1,
            "batch_norm": True,
            "layer_norm": False,
            "early_stopping_enabled": False,
        },
        "time_series": {
            "dataset_csv": tmp_time_series_dataset.name,  # "time_series_example.csv"
            "datetime_column": "timestamp",
            "target_column": "target",
            "value_columns": ["feat1", "feat2"],
            "val_fraction": 0.2,
            "input_window": 3,
            "prediction_horizon": 1,
            "step_size": 1,
            "task_type": "regression",
        },
    }

    config_path = tmp_path / "time_series_config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
