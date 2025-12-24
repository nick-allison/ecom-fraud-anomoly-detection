from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ecom_fraud.exceptions import ConfigError

# ----------------------------------------------------------------------
# Environment + discovery defaults
# ----------------------------------------------------------------------

# All environment variables for this project will be prefixed with this.
# Example: ML_TABULAR_ENV, ML_TABULAR_CONFIG_PATH, ML_TABULAR_TRAINING__TEST_SIZE, etc.
ENV_PREFIX = "ML_TABULAR_"
ENV_ENV_NAME = f"{ENV_PREFIX}ENV"  # ML_TABULAR_ENV
ENV_CONFIG_PATH = f"{ENV_PREFIX}CONFIG_PATH"

DEFAULT_ENV = os.getenv(ENV_ENV_NAME, "dev").lower()
DEFAULT_CONFIG_FILENAMES = ("config.yaml", "config.yml")


# ----------------------------------------------------------------------
# Section models
# ----------------------------------------------------------------------


class PathsConfig(BaseModel):
    """Filesystem-related configuration: where data and models live."""

    model_config = ConfigDict(frozen=True)

    base_dir: Path = Path(".")
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    processed_dir: Path = Path("data/processed")
    models_dir: Path = Path("models")

    def resolve(self, base: Path | None = None) -> PathsConfig:
        """Return a copy of this config with all paths made absolute."""
        base_dir = Path(base) if base is not None else self.base_dir
        return PathsConfig(
            base_dir=base_dir,
            data_dir=(base_dir / self.data_dir).resolve(),
            raw_dir=(base_dir / self.raw_dir).resolve(),
            processed_dir=(base_dir / self.processed_dir).resolve(),
            models_dir=(base_dir / self.models_dir).resolve(),
        )


class TrainingConfig(BaseModel):
    """Generic configuration for model training and evaluation."""

    model_config = ConfigDict(frozen=True)

    random_seed: int = Field(42, ge=0, description="Random seed for reproducibility.")
    test_size: float = Field(
        0.2,
        ge=0.0,
        le=0.9,
        description="Fraction of data reserved for validation/test.",
    )
    n_estimators: int = Field(
        100,
        gt=0,
        description="Example hyperparameter for tree-based models.",
    )
    learning_rate: float = Field(
        0.1,
        gt=0.0,
        description="Example hyperparameter for gradient-based models.",
    )


class DatabaseConfig(BaseModel):
    """Configuration for relational database access."""

    model_config = ConfigDict(frozen=True)

    url: str = Field(
        "sqlite:///example.db",
        description=(
            "SQLAlchemy-style URL, e.g. "
            "'postgresql+psycopg://user:pass@host:5432/dbname'. "
            "Override via YAML or environment variables in real projects."
        ),
    )
    schema: str | None = Field(
        None,
        description="Default schema name (optional, database-dependent).",
    )
    echo: bool = Field(
        False,
        description="If True, echo SQL statements (useful in dev; off in prod).",
    )


class MongoConfig(BaseModel):
    """Configuration for MongoDB access."""

    model_config = ConfigDict(frozen=True)

    uri: str = Field(
        "mongodb://localhost:27017",
        description=(
            "MongoDB connection URI. May include credentials; do not log this "
            "directly in application logs."
        ),
    )
    database: str = Field(
        "ml_tabular",
        description="Default MongoDB database name.",
    )


class MlflowConfig(BaseModel):
    """Configuration for MLflow tracking (optional)."""

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(
        False,
        description="If True, training scripts may log runs/metrics to MLflow.",
    )
    tracking_uri: str | None = Field(
        None,
        description=(
            "MLflow tracking URI. If None, MLflow's default behavior is used "
            "(e.g. local ./mlruns)."
        ),
    )
    experiment_name: str | None = Field(
        None,
        description="Default MLflow experiment name; scripts may override this per run.",
    )
    run_name: str | None = Field(
        None,
        description="Optional MLflow run name template or default.",
    )
    log_artifacts: bool = Field(
        True,
        description="If True, training scripts may log artifacts (models, configs, etc.) to MLflow.",
    )


class KaggleConfig(BaseModel):
    """Configuration for Kaggle integration (optional)."""

    model_config = ConfigDict(frozen=True)

    dataset: str | None = Field(
        None,
        description=(
            "Default Kaggle dataset identifier (e.g. 'zusmani/metro-interstate-traffic-volume'). "
            "Training scripts or utilities may use this if no explicit dataset is provided."
        ),
    )
    competition: str | None = Field(
        None,
        description=(
            "Default Kaggle competition identifier (e.g. 'titanic'). "
            "Used when working with competition data instead of datasets."
        ),
    )
    download_subdir: str = Field(
        "kaggle",
        description=(
            "Subdirectory under paths.raw_dir where Kaggle downloads should be stored "
            "(e.g. data/raw/kaggle)."
        ),
    )


class TimeSeriesConfig(BaseModel):
    """Generic configuration hints for time-series experiments.

    This is not required for tabular-only runs, but can be used by time-series
    training scripts to interpret YAML settings like those in train_ts_baseline.yaml.
    """

    model_config = ConfigDict(frozen=True)

    datetime_column: str = Field(
        "timestamp",
        description="Name of the datetime column in time-series datasets.",
    )
    target_column: str = Field(
        "target",
        description="Name of the target column for forecasting.",
    )
    group_column: str | None = Field(
        None,
        description=(
            "Optional ID column for multiple related series (e.g. 'store_id'). "
            "If None, data is treated as a single global series."
        ),
    )
    freq: str | None = Field(
        "D",
        description="Nominal frequency of the time series (e.g. 'D', 'H'). Used as a hint.",
    )
    input_window: int = Field(
        30,
        gt=0,
        description="Number of past time steps per input window for sequence models.",
    )
    prediction_horizon: int = Field(
        7,
        gt=0,
        description="Number of steps ahead to predict (forecast horizon).",
    )
    max_series_length: int | None = Field(
        None,
        description="Optional cap on series length for training. None = use full history.",
    )
    normalize_per_series: bool = Field(
        True,
        description="Whether to normalize each series individually (per-series scaling).",
    )


# ----------------------------------------------------------------------
# Top-level settings
# ----------------------------------------------------------------------


class AppConfig(BaseSettings):
    """Top-level application configuration.

    This reads values from (in order of precedence):

    1. Keyword arguments passed to the constructor (e.g. from a YAML file).
    2. Environment variables (prefixed with ML_TABULAR_).
    3. A .env file (if present).
    4. Default values declared in the model fields.

    Nested fields (like `paths`, `training`, `database`, `mongo`, `mlflow`,
    `kaggle`, `time_series`) can be overridden via environment variables using
    the `env_nested_delimiter`:

        ML_TABULAR_TRAINING__N_ESTIMATORS=200
        ML_TABULAR_PATHS__DATA_DIR=/mnt/data
        ML_TABULAR_DATABASE__URL=postgresql+psycopg://user:pass@host/db
        ML_TABULAR_MONGO__URI=mongodb://user:pass@host:27017
        ML_TABULAR_MLFLOW__ENABLED=true
        ML_TABULAR_KAGGLE__DATASET=zusmani/metro-interstate-traffic-volume

    A YAML config file can define either:

    - A flat config:

        env: "prod"
        log_level: "INFO"
        paths:
          data_dir: "data"
        training:
          test_size: 0.3
        database:
          url: "postgresql+psycopg://..."
        mlflow:
          enabled: true
          experiment_name: "tabular_baselines"

    - Or environment-specific profiles:

        dev:
          env: "dev"
          log_level: "DEBUG"
        prod:
          env: "prod"
          log_level: "INFO"
          database:
            url: "postgresql+psycopg://..."
          time_series:
            input_window: 60
            prediction_horizon: 14

    In the second case, the `ML_TABULAR_ENV` value (or DEFAULT_ENV)
    selects which profile to load.
    """

    model_config = SettingsConfigDict(
        env_prefix=ENV_PREFIX,
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        frozen=True,  # make config immutable
    )

    # Private metadata (not exposed in model_dump by default)
    _source_path: Path | None = PrivateAttr(default=None)
    _loaded_env: str | None = PrivateAttr(default=None)

    env: str = Field(
        "dev",
        description="Environment name, e.g. 'dev', 'prod', 'test'.",
    )
    log_level: str = Field(
        "INFO",
        description="Default log level for the application.",
    )
    experiment_name: str = Field(
        "default_experiment",
        description="Human-readable name for the current experiment/run.",
    )

    paths: PathsConfig = PathsConfig()
    training: TrainingConfig = TrainingConfig()
    database: DatabaseConfig = DatabaseConfig()
    mongo: MongoConfig = MongoConfig()
    mlflow: MlflowConfig = MlflowConfig()
    kaggle: KaggleConfig = KaggleConfig()
    time_series: TimeSeriesConfig = TimeSeriesConfig()

    @model_validator(mode="after")
    def _check_invariants(self) -> AppConfig:
        """Cross-field validation and invariants."""
        # Avoid debug logging in explicit production envs.
        if self.env.lower() in {"prod", "production"} and self.log_level.upper() == "DEBUG":
            raise ValueError(
                "In production environment, log_level should not be DEBUG. "
                "Use INFO or higher."
            )

        # Avoid accidentally writing raw and processed data to the same directory.
        if self.paths.raw_dir == self.paths.processed_dir:
            raise ValueError("paths.raw_dir and paths.processed_dir must be different.")

        return self

    def resolved_paths(self) -> PathsConfig:
        """Return a PathsConfig with all paths fully resolved."""
        return self.paths.resolve(self.paths.base_dir)

    def to_dict(self, *, include_private: bool = False) -> dict[str, Any]:
        """Return a plain dict representation of the effective configuration."""
        data = self.model_dump()
        if include_private:
            data["_source_path"] = str(self._source_path) if self._source_path else None
            data["_loaded_env"] = self._loaded_env
        return data

    def to_yaml(self, path: Path | str, *, include_private: bool = False) -> None:
        """Write the effective configuration to a YAML file.

        Useful for experiment tracking and reproducibility: the exact config
        used for a run can be persisted alongside model artifacts.
        """
        target = Path(path)
        dump_data = self.to_dict(include_private=include_private)
        target.write_text(
            yaml.safe_dump(dump_data, sort_keys=False),
            encoding="utf-8",
        )


# ----------------------------------------------------------------------
# Loading + caching
# ----------------------------------------------------------------------

_config_cache: AppConfig | None = None


def _discover_default_config_path() -> Path | None:
    """Return a default config path if one can be discovered.

    Priority:
    1. ML_TABULAR_CONFIG_PATH environment variable (if set and exists).
    2. ./config.yaml or ./config.yml in the current working directory.
    3. None, if nothing is found (env + defaults will be used).
    """
    env_path = os.getenv(ENV_CONFIG_PATH)
    if env_path:
        candidate = Path(env_path)
        if candidate.exists():
            return candidate
        # If user explicitly set ML_TABULAR_CONFIG_PATH and it does not exist,
        # we'll let load_config raise a ConfigError later.

    cwd = Path.cwd()
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = cwd / name
        if candidate.exists():
            return candidate

    return None


def _select_profile_from_yaml(
    loaded: Mapping[str, Any],
    effective_env: str,
) -> Mapping[str, Any]:
    """Given loaded YAML and an env, decide whether to use profiles or a flat config.

    If `loaded` has a key matching `effective_env` whose value is a mapping,
    that subtree is treated as the active profile. Otherwise, `loaded` itself
    is treated as the config mapping.
    """
    section = loaded.get(effective_env)
    if isinstance(section, Mapping):
        return section
    return loaded


def load_config(
    config_path: str | Path | None = None,
    *,
    env: str | None = None,
) -> AppConfig:
    """Load and validate application configuration.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file. If omitted, attempts to discover a
        config file using ML_TABULAR_CONFIG_PATH or ./config.yaml / ./config.yml.
    env:
        Optional environment name used to select YAML profile (e.g. 'dev', 'prod').
        If omitted, ML_TABULAR_ENV (or 'dev') is used.

    Raises
    ------
    ConfigError
        If the config file does not exist, cannot be parsed, or fails validation.
    """
    effective_env = (env or DEFAULT_ENV).lower()
    config_data: dict[str, Any] = {}

    # Discover config path if not explicitly provided
    path: Path | None
    if config_path is not None:
        path = Path(config_path)
    else:
        path = _discover_default_config_path()

    if path is not None:
        if not path.exists():
            raise ConfigError(
                f"Config file not found: {path}",
                code="config_file_not_found",
                context={"config_path": str(path), "env": effective_env},
                location="ml_tabular.config.load_config",
            )

        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ConfigError(
                f"Failed to read config file: {path}",
                code="config_read_error",
                cause=exc,
                context={"config_path": str(path), "env": effective_env},
                location="ml_tabular.config.load_config",
            ) from exc

        try:
            loaded = yaml.safe_load(text) or {}
        except Exception as exc:
            raise ConfigError(
                f"Failed to parse YAML config file: {path}",
                code="config_parse_error",
                cause=exc,
                context={"config_path": str(path), "env": effective_env},
                location="ml_tabular.config.load_config",
            ) from exc

        if not isinstance(loaded, Mapping):
            raise ConfigError(
                f"Top-level config in {path} must be a mapping/object, got {type(loaded)}",
                code="config_structure_error",
                context={"config_path": str(path), "env": effective_env},
                location="ml_tabular.config.load_config",
            )

        # Support environment profiles: dev/prod/test sections in YAML.
        profile = _select_profile_from_yaml(loaded, effective_env)
        if not isinstance(profile, Mapping):
            raise ConfigError(
                f"Selected config profile for env='{effective_env}' in {path} "
                f"must be a mapping/object, got {type(profile)}",
                code="config_profile_error",
                context={"config_path": str(path), "env": effective_env},
                location="ml_tabular.config.load_config",
            )

        config_data.update(dict(profile))

    try:
        cfg = AppConfig(**config_data)
    except ValidationError as exc:
        raise ConfigError(
            "Configuration validation failed",
            code="config_validation_error",
            cause=exc,
            context={
                "config_path": str(path) if path is not None else None,
                "env": effective_env,
                "errors": exc.errors(),
            },
            location="ml_tabular.config.load_config",
        ) from exc

    # Attach metadata about where this config came from
    cfg._source_path = path
    cfg._loaded_env = effective_env

    return cfg


def get_config(
    config_path: str | Path | None = None,
    *,
    env: str | None = None,
    force_reload: bool = False,
) -> AppConfig:
    """Return the cached AppConfig instance, loading it if necessary.

    Parameters
    ----------
    config_path:
        Optional path to a YAML config file. Only used on first load or when
        `force_reload` is True. If omitted, config discovery is used.
    env:
        Optional environment name (e.g. 'dev', 'prod') for profile selection.
        If omitted, ML_TABULAR_ENV (or 'dev') is used.
    force_reload:
        If True, reload the configuration even if it was cached.

    Notes
    -----
    This function is the main entrypoint that other modules should use:

        from ml_tabular.config import get_config

        cfg = get_config()
        paths = cfg.resolved_paths()
    """
    global _config_cache

    if _config_cache is None or force_reload:
        _config_cache = load_config(config_path=config_path, env=env)

    return _config_cache


def get_paths(
    config_path: str | Path | None = None,
    *,
    env: str | None = None,
    force_reload: bool = False,
) -> PathsConfig:
    """Convenience helper to get resolved PathsConfig directly."""
    cfg = get_config(config_path=config_path, env=env, force_reload=force_reload)
    return cfg.resolved_paths()
