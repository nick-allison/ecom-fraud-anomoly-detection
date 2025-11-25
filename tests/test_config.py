from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml_tabular.config import AppConfig, get_config, get_paths
from ml_tabular.exceptions import ConfigError


# ---------------------------------------------------------------------------
# Basic loading from explicit config paths
# ---------------------------------------------------------------------------


def test_load_tabular_config_from_path(tabular_config_path: Path) -> None:
    """AppConfig should load cleanly from a tabular YAML config."""
    cfg = get_config(config_path=tabular_config_path, force_reload=True)
    assert isinstance(cfg, AppConfig)

    # Top-level fields
    assert cfg.env == "dev"
    assert cfg.experiment_name == "test_tabular"
    assert cfg.log_level.upper() == "INFO"

    # Training section matches what conftest wrote
    assert cfg.training.batch_size == 16
    assert cfg.training.num_epochs == 3
    assert cfg.training.learning_rate == pytest.approx(1e-3)

    # Tabular section should be present and correctly wired
    assert hasattr(cfg, "tabular")
    assert cfg.tabular.dataset_csv == "tabular_example.csv"
    assert cfg.tabular.target_column == "target"
    assert cfg.tabular.feature_columns == ["feat1", "feat2"]


def test_load_time_series_config_from_path(time_series_config_path: Path) -> None:
    """AppConfig should load cleanly from a time-series YAML config."""
    cfg = get_config(config_path=time_series_config_path, force_reload=True)
    assert isinstance(cfg, AppConfig)

    assert cfg.env == "dev"
    assert cfg.experiment_name == "test_ts"
    assert cfg.log_level.upper() == "INFO"

    assert cfg.training.batch_size == 8
    assert cfg.training.num_epochs == 3
    assert cfg.training.learning_rate == pytest.approx(1e-3)

    # Time-series specific section
    assert hasattr(cfg, "time_series")
    ts = cfg.time_series
    assert ts.dataset_csv == "time_series_example.csv"
    assert ts.datetime_column == "timestamp"
    assert ts.target_column == "target"
    assert ts.value_columns == ["feat1", "feat2"]
    assert ts.input_window == 3
    assert ts.prediction_horizon == 1
    assert ts.step_size == 1


# ---------------------------------------------------------------------------
# Paths resolution
# ---------------------------------------------------------------------------


def test_get_paths_resolves_directories(tabular_config_path: Path) -> None:
    """get_paths should resolve absolute directories and they should exist."""
    paths = get_paths(config_path=tabular_config_path, force_reload=True)

    # These directories were created by the tabular_config_path fixture
    assert paths.data_dir.exists()
    assert paths.raw_dir.exists()
    assert paths.processed_dir.exists()
    assert paths.models_dir.exists()

    # The dataset file referenced in the tabular config should exist under data_dir
    dataset_path = paths.data_dir / "tabular_example.csv"
    assert dataset_path.exists()


def test_get_paths_for_time_series(time_series_config_path: Path) -> None:
    """Same as above, but for the time-series config."""
    paths = get_paths(config_path=time_series_config_path, force_reload=True)

    assert paths.data_dir.exists()
    assert paths.models_dir.exists()

    dataset_path = paths.data_dir / "time_series_example.csv"
    assert dataset_path.exists()


# ---------------------------------------------------------------------------
# Environment variable overrides (nested / config path discovery)
# ---------------------------------------------------------------------------


def test_env_var_overrides_training_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tabular_config_path: Path,
) -> None:
    """ML_TEMPLATE_TRAINING__BATCH_SIZE should override batch_size from YAML."""
    monkeypatch.setenv("ML_TEMPLATE_TRAINING__BATCH_SIZE", "64")

    cfg = get_config(config_path=tabular_config_path, force_reload=True)
    assert cfg.training.batch_size == 64


def test_config_path_discovery_via_env(
    monkeypatch: pytest.MonkeyPatch,
    tabular_config_path: Path,
) -> None:
    """If ML_TEMPLATE_CONFIG_PATH is set, get_config(None) should use it."""
    monkeypatch.setenv("ML_TEMPLATE_CONFIG_PATH", str(tabular_config_path))

    # Calling without config_path should trigger discovery logic
    cfg = get_config(config_path=None, force_reload=True)

    assert cfg.env == "dev"
    assert cfg.experiment_name == "test_tabular"
    assert cfg.training.batch_size == 16


# ---------------------------------------------------------------------------
# Profile-based configs (dev / prod sections)
# ---------------------------------------------------------------------------


def test_profile_selection_with_env_arg(
    tmp_path: Path,
    tabular_config_path: Path,
) -> None:
    """When a config file has dev/prod profiles, get_config(env=...) should select correctly."""
    # Start from the base config created by the fixture
    base_cfg_data = yaml.safe_load(tabular_config_path.read_text(encoding="utf-8"))

    # Build a profiled config: {dev: {...}, prod: {...}}
    dev_cfg = base_cfg_data
    prod_cfg = {
        **base_cfg_data,
        "experiment_name": "prod_tabular",
        "training": {
            **base_cfg_data["training"],
            "batch_size": 128,
        },
    }

    profiled = {"dev": dev_cfg, "prod": prod_cfg}

    profiled_path = tmp_path / "profiled_tabular.yaml"
    profiled_path.write_text(yaml.safe_dump(profiled, sort_keys=False), encoding="utf-8")

    # Explicitly pick the prod profile
    cfg_prod = get_config(config_path=profiled_path, env="prod", force_reload=True)
    assert cfg_prod.env == "prod"
    assert cfg_prod.experiment_name == "prod_tabular"
    assert cfg_prod.training.batch_size == 128

    # Default dev profile
    cfg_dev = get_config(config_path=profiled_path, env="dev", force_reload=True)
    assert cfg_dev.env == "dev"
    assert cfg_dev.experiment_name == base_cfg_data["experiment_name"]
    assert cfg_dev.training.batch_size == base_cfg_data["training"]["batch_size"]


# ---------------------------------------------------------------------------
# Validation failures and ConfigError
# ---------------------------------------------------------------------------


def test_invalid_config_raises_config_error(
    tmp_path: Path,
    tabular_config_path: Path,
) -> None:
    """Schema/type violations in YAML should surface as ConfigError."""
    base_cfg_data = yaml.safe_load(tabular_config_path.read_text(encoding="utf-8"))

    # Introduce an invalid type: num_epochs should be an int, we make it a string
    bad_cfg_data = {
        **base_cfg_data,
        "training": {
            **base_cfg_data["training"],
            "num_epochs": "not-an-int",
        },
    }

    bad_path = tmp_path / "bad_tabular_config.yaml"
    bad_path.write_text(yaml.safe_dump(bad_cfg_data, sort_keys=False), encoding="utf-8")

    with pytest.raises(ConfigError) as ctx:
        get_config(config_path=bad_path, force_reload=True)

    exc = ctx.value
    # Basic sanity: it's our custom ConfigError, not just a raw ValidationError
    assert isinstance(exc, ConfigError)

    # If the ConfigError exposes a code, it's nice if it mentions validation
    code = getattr(exc, "code", "")
    if code:
        assert "validation" in code.lower()
