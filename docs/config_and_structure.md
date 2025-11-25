# Configuration and Project Structure

This document explains how configuration flows through the `ml_tabular` template and how the repository is structured. The goal is to make it easy to:

- Understand where to put configuration (code vs YAML vs environment)
- Navigate the directory layout
- Extend the template for real tabular and time-series projects
- Keep everything reproducible and maintainable


## 1. Goals and guiding principles

The template is designed to:

1) Separate **configuration** from **code**  
   - Behavior is driven by YAML files and environment variables, not hard-coded constants.
   - You can run the same code in different environments (dev/prod) with different settings.

2) Provide a **clear, layered structure**  
   - Data access (files, SQL, Mongo, Kaggle)
   - Feature engineering (tabular + time-series)
   - Modeling and training (PyTorch)
   - Experiment tracking (MLflow)
   - Command-line interface and notebooks for experimentation

3) Support both **tabular** and **time-series** workflows from the same core framework.

4) Be **portfolio-ready**  
   - Organized like a “real” internal template you might build for a team.
   - Clean enough that a reviewer can quickly understand how you think about ML projects.


## 2. Configuration overview: three main layers

There are three main configuration layers:

1) **Static project configuration** (Python packaging and tooling)
   - pyproject.toml
   - .pre-commit-config.yaml
   - .editorconfig
   - .gitignore

2) **Runtime configuration model** (validated in Python)
   - src/ml_tabular/config.py

3) **Experiment- and environment-specific settings**
   - configs/*.yaml (e.g. train_tabular_baseline.yaml, train_ts_baseline.yaml)
   - Environment variables (e.g. ML_TABULAR_ENV, ML_TABULAR_CONFIG_PATH)

Together, these support both local experimentation and more production-like usage.


## 3. Static project configuration

### 3.1 pyproject.toml

pyproject.toml defines:

- The project’s identity:
  - Name, version, description, authors, URLs
- Build configuration (setuptools)
- Runtime dependencies:
  - numpy, pandas, scikit-learn, torch, pydantic, pydantic-settings, pyyaml
- Optional extras:
  - dev      : pytest, mypy, ruff, pre-commit, jupyter, ipykernel
  - docs     : mkdocs, mkdocs-material
  - mlops    : mlflow, kaggle
  - validation : pandera
- Tooling configuration:
  - ruff (lint + format)
  - mypy (static typing)
  - pytest (test discovery, coverage)

Key points:

- `requires-python = ">=3.10,<3.13"` keeps you in a stable band where dependencies (torch, etc.) are well-supported.
- `include = ["ml_tabular*"]` in setuptools ensures the `ml_tabular` package (and its submodules) are included.
- Tool sections keep project-wide standards in one place (formatter, linter, type checker, tests).

When installed via:

- `pip install -e ".[dev,mlops,validation]"`

you get both core functionality and the optional integrations.


### 3.2 .editorconfig

.editorconfig enforces consistent formatting across editors:

- Global defaults:
  - UTF-8 encoding
  - LF line endings
  - Final newline at end of file
  - Spaces (not tabs) with indent size 4 by default
  - Trim trailing whitespace, except where explicitly allowed

- Language-specific tweaks:
  - Python: 4-space indent
  - YAML/TOML/JSON: 2-space indent
  - Markdown: allow trailing spaces (for hard line breaks)
  - Makefile: tabs (Makefiles require real tabs)

This keeps the codebase consistent regardless of IDE/editor.


### 3.3 .pre-commit-config.yaml

.pre-commit-config.yaml wires pre-commit hooks:

- Ruff:
  - `ruff` (lint, with `--fix`)
  - `ruff-format` (formatter)
- mypy:
  - type checking using the config in pyproject.toml
- pre-commit-hooks:
  - check-yaml
  - end-of-file-fixer
  - trailing-whitespace
  - check-merge-conflict
  - check-added-large-files

There is also an optional local `pytest` hook (manual stage) so you can easily run tests via pre-commit if desired.

Installing hooks once (`pre-commit install`) keeps the repo clean and consistent.


### 3.4 .gitignore

.gitignore combines:

- Standard Python ignores:
  - __pycache__/
  - *.pyc
  - build/dist/egg artifacts
  - .tox, .pytest_cache, .mypy_cache, etc.
- Environment and IDE:
  - .env, .venv, env/, venv/
  - .DS_Store
- ML-specific:
  - data/raw/
  - data/interim/
  - data/processed/
  - models/
  - checkpoints/
  - logs/
  - *.log

This keeps large and environment-specific artifacts out of version control. Only code, configs, and docs are tracked.


## 4. Runtime configuration: AppConfig and friends

### 4.1 src/ml_tabular/config.py

This module is the heart of runtime configuration. It:

- Defines strongly-typed configuration sections using Pydantic models:
  - PathsConfig
  - TrainingConfig
  - DatabaseConfig
  - MongoConfig
  - MlflowConfig
  - KaggleConfig
  - TimeSeriesConfig
- Wraps these inside a single top-level AppConfig (extends BaseSettings).
- Implements robust loading and validation from:
  - YAML files
  - Environment variables
  - Default values

Key patterns:

- `ENV_PREFIX = "ML_TABULAR_"`  
  All environment variables for this project start with `ML_TABULAR_`.
  Examples:
  - ML_TABULAR_ENV=prod
  - ML_TABULAR_CONFIG_PATH=/path/to/config.yaml
  - ML_TABULAR_TRAINING__TEST_SIZE=0.3

- AppConfig.model_config includes:
  - `env_prefix = "ML_TABULAR_"`
  - `env_nested_delimiter = "__"`
  - `env_file = ".env"`
  - `extra = "ignore"` (ignores unknown fields)
  - `frozen = True` (immutable config object)

This lets you override nested config fields via environment variables, while still validating everything using Pydantic.


### 4.2 Config sections (high level)

- PathsConfig:
  - base_dir, data_dir, raw_dir, processed_dir, models_dir
  - `resolve()` method converts relative paths to absolute.

- TrainingConfig:
  - Generic hyperparameters (random_seed, test_size, etc.)
  - Primarily used by higher-level scripts for consistent defaults.

- DatabaseConfig:
  - url, schema, echo
  - Used by `ml_tabular.data.sql` to create SQLAlchemy engines safely.

- MongoConfig:
  - uri, database
  - Used by `ml_tabular.data.mongodb` (optional dependency on pymongo).

- MlflowConfig:
  - enabled, tracking_uri, experiment_name, run_name, log_artifacts
  - Controls whether and how MLflow logging is performed.

- KaggleConfig:
  - dataset, competition, download_subdir
  - Used by `ml_tabular.data.kaggle` to pull data programmatically.

- TimeSeriesConfig:
  - datetime_column, target_column, group_column, freq
  - input_window, prediction_horizon, max_series_length, normalize_per_series
  - Provides global hints for time-series training scripts.


### 4.3 Config loading and caching

Key functions:

- `load_config(config_path: str | Path | None = None, env: str | None = None) -> AppConfig`
  - Reads YAML, applies environment profile, validates using AppConfig.
  - On error, raises ConfigError with rich context.

- `get_config(config_path: str | Path | None = None, env: str | None = None, force_reload: bool = False) -> AppConfig`
  - Cached accessor; use this everywhere instead of constructing AppConfig directly.

- `get_paths(...) -> PathsConfig`
  - Convenience helper returning resolved PathsConfig.

Default discovery rules:

1) If ML_TABULAR_CONFIG_PATH is set and points to an existing file, use it.
2) Else, look for ./config.yaml or ./config.yml in the current working directory.
3) Else, fall back to defaults (config object with default values, no file).


### 4.4 YAML environment profiles

YAML configs can be:

- Flat:
  - Keys correspond directly to AppConfig fields.
  - Example: env, log_level, paths, training, mlflow, etc.

- Profile-based:
  - Top-level keys represent environments (dev, prod, test).
  - AppConfig selects the profile based on ML_TABULAR_ENV or DEFAULT_ENV="dev".
  - Example:
    - dev: { ... }
    - prod: { ... }

This lets you maintain a single config.yaml with multiple environment sections if you want.


## 5. Repository structure

The high-level directory layout is:

- pyproject.toml
- .pre-commit-config.yaml
- .editorconfig
- .gitignore
- configs/
- docs/
- notebooks/
- src/
  - ml_tabular/
    - __init__.py
    - config.py
    - exceptions.py
    - logging_config.py
    - data/
    - features/
    - torch/
    - sklearn_utils/
    - mlops/
    - cli/
- tests/
  - (mirrors ml_tabular structure)


### 5.1 configs/

This directory holds YAML configs:

- train_tabular_baseline.yaml
  - File names (train/valid/test)
  - Target and feature columns
  - Basic model and training hyperparameters
  - MLflow options

- train_ts_baseline.yaml
  - Time-series-specific settings:
    - datetime_column, target_column, group_column
    - window sizes and horizon
    - covariates, etc.
  - Model and training hyperparameters

You can add additional configs like:

- train_tabular_mlp_deep.yaml
- train_ts_transformer.yaml


### 5.2 docs/

Contains Markdown documentation:

- index.md:
  - High-level overview, motivation, and capabilities.

- quickstart_tabular.md:
  - End-to-end tabular workflow: load data, build features, train MLP, track with MLflow.

- quickstart_time_series.md:
  - End-to-end time-series workflow: sliding windows, time features, MLP over sequences.

- config_and_structure.md (this file):
  - Detailed explanation of configuration and repo structure.

These can be used as plain Markdown or wired into a MkDocs site.


### 5.3 notebooks/

Notebooks give executable examples:

- 00_tabular_quickstart.ipynb
  - First tabular experiment.

- 01_tabular_eda_and_features.ipynb
  - EDA and hands-on feature engineering for tabular data.

- 10_time_series_quickstart.ipynb
  - First time-series experiment.

- 11_time_series_eda_and_windows.ipynb
  - EDA + visualization of time-series windows and features.

- 90_dev_scratchpad.ipynb
  - General playground for trying new ideas without affecting the main code.


## 6. src/ml_tabular package structure

### 6.1 Core infrastructure

- `src/ml_tabular/__init__.py`
  - Defines package exports, e.g. get_config, get_logger, etc.

- `src/ml_tabular/config.py`
  - Runtime configuration model and loader (described above).

- `src/ml_tabular/exceptions.py`
  - Structured error classes:
    - AppError (base)
    - ConfigError
    - DataError
    - ModelError
    - PipelineError
  - Includes `to_dict()` for structured logging/serialization.

- `src/ml_tabular/logging_config.py`
  - Central logging setup:
    - Supports env-driven level, directory, environment (dev/prod), and format (text/json).
    - Optionally uses python-json-logger for JSON logs.
    - Configures console and rotating file handlers.
  - `get_logger(name)` ensures logging is configured lazily.


### 6.2 Data layer: src/ml_tabular/data/

Responsible for getting data into pandas (or similar) from various sources.

Key modules:

- `loading.py`
  - Load CSV/Parquet/Feather into DataFrames.
  - Validate existence and required columns.
  - Log shape and format.
  - Helper functions:
    - load_raw_dataset()
    - load_processed_dataset()

- `sql.py`
  - SQLAlchemy integration:
    - get_engine() based on AppConfig.database
    - load_sql_query() and load_sql_table()
    - execute_sql() for non-SELECT statements
  - Safely logs only driver and query previews; never logs full connection strings.

- `mongodb.py`
  - MongoDB integration (optional, contingent on pymongo).
  - get_mongo_client() and get_collection()
  - load_mongo_collection() and count_documents()
  - Safely logs only scheme/host/port, not credentials.

- `kaggle.py`
  - Kaggle API integration (optional).
  - Download datasets or competition data into data/raw/kaggle/.
  - Hooks into AppConfig.kaggle for default dataset/competition identifiers.


### 6.3 Feature layer: src/ml_tabular/features/

Applies **stateless**, per-row transformations to DataFrames.

- `tabular.py`
  - Features common for tabular data:
    - Datetime expansion (year, month, day, dayofweek, hour, weekofyear, is_weekend, etc.)
    - log1p transforms (for count-like variables)
    - Ratio features
    - Power features (e.g. squared terms)
    - Interaction features (products of columns)
  - Uses FeatureSpec to declare what to do:
    - datetime_columns
    - log1p_columns
    - ratio_features
    - power_features
    - interaction_features
  - build_features() applies the spec and returns a new DataFrame.

- `time_series.py`
  - Time-series-specific feature engineering:
    - Derived date attributes for forecasting (similar to tabular, but oriented to sequence modeling).
    - Helpers for ensuring consistent ordering and alignment before windowing.
  - Works hand-in-hand with the torch time-series dataset.


### 6.4 Modeling and training: src/ml_tabular/torch/

Deep-learning components for both tabular and time-series.

- `torch/datasets/tabular.py`
  - TabularDataset:
    - Wraps a pandas DataFrame with:
      - feature columns
      - target column(s)
      - optional sample weights
    - Produces tensors suitable for feeding into MLPs or other feed-forward models.

- `torch/datasets/time_series.py`
  - TimeSeriesDataset:
    - Converts “long” time-series DataFrames into sliding windows.
    - Handles:
      - grouping by series_id (if specified)
      - sorting by datetime
      - normalization per series (if desired)
      - constructing X windows and y horizons as tensors

- `torch/models/tabular_mlp.py`
  - A flexible MLP model class:
    - Arbitrary input dimension, hidden sizes, activation, dropout.
    - Supports regression and classification outputs.
    - Used for both tabular and flattened time-series windows.

- `torch/training/loops.py`
  - Concise training and evaluation loops:
    - train_epoch()
    - evaluate()
  - Accepts DataLoaders, model, optimizer, loss function, device, metrics, etc.
  - Designed to be reused by both tabular and time-series scripts.


### 6.5 Scikit-learn utilities: src/ml_tabular/sklearn_utils/

Provides helper utilities to leverage sklearn where it’s strongest:

- `tabular_pipeline.py`
  - Constructs sklearn-like preprocessing pipelines for:
    - imputers
    - scalers
    - encoders (e.g. OneHotEncoder)
  - These pipelines can:
    - Generate numpy arrays ready for PyTorch datasets
    - Or be used directly with sklearn models

This allows you to apply exactly the same preprocessing for both shallow models (sklearn) and deep models (torch), or at least keep them aligned.


### 6.6 MLOps utilities: src/ml_tabular/mlops/

Integrates MLflow in a structured way.

- `mlflow_utils.py`
  - High-level helpers to:
    - Set the tracking URI
    - Ensure experiments exist
    - Start and end runs safely
    - Log parameters, metrics, and artifacts
  - Used inside training scripts or CLI commands to keep MLflow logic consistent and centralized.


### 6.7 CLI layer: src/ml_tabular/cli/

Provides a user-friendly entrypoint for running experiments:

- `cli.py` (or `app.py`, depending on your naming)
  - Built with Typer or click.
  - Commands might include:
    - `train-tabular`
    - `train-ts`
    - `evaluate`
  - Each command:
    - Accepts a `--config` path
    - Uses get_config() and YAML to configure the run
    - Calls into orchestration logic (load data, features, datasets, model, training loops)


## 7. Tests: tests/ directory

The test suite mirrors the src layout:

- tests/conftest.py
  - Common fixtures for temporary directories, tiny DataFrames, etc.

- tests/test_exceptions.py
  - Ensures AppError and subclasses behave and serialize as expected.

- tests/test_logging_config.py
  - Verifies logging is configured without crashing.
  - Checks that JSON vs text formatting decisions are respected.

- tests/test_config.py
  - Ensures AppConfig:
    - Validates fields correctly
    - Handles environment profiles
    - Respects environment variables

- tests/torch/datasets/test_tabular_dataset.py
  - Checks TabularDataset shapes and types.

- tests/torch/datasets/test_time_series_dataset.py
  - Ensures windows and targets line up correctly.

- tests/torch/models/test_tabular_mlp.py
  - Sanity tests for model forward pass and dimensions.

- tests/torch/training/test_training_loops.py
  - Basic smoke tests for train/evaluate loops.

- tests/cli/test_cli.py
  - Ensures CLI commands parse arguments and call underlying code.

- tests/mlops/test_mlflow_utils.py
  - Verifies MLflow helpers can be imported and called (often with mocks).


## 8. Putting it all together

In a typical workflow:

1) You define your config in:
   - configs/train_tabular_baseline.yaml
   - configs/train_ts_baseline.yaml
   - or a new config you create.

2) You optionally set environment variables:
   - ML_TABULAR_ENV=dev or prod
   - ML_TABULAR_CONFIG_PATH=./configs/some_profile.yaml

3) You run:
   - `python train_tabular_mlp.py --config configs/train_tabular_baseline.yaml`
   - `python train_ts_mlp.py --config configs/train_ts_baseline.yaml`
   - or equivalent CLI commands.

4) The scripts:
   - Configure logging
   - Load AppConfig and the YAML config
   - Load data via ml_tabular.data (files, SQL, Mongo, Kaggle)
   - Apply feature engineering via ml_tabular.features
   - Build PyTorch datasets and models
   - Train using ml_tabular.torch.training.loops
   - Optionally log everything to MLflow

5) The repository remains:
   - Well-structured
   - Testable
   - Easy to extend with new models or data sources
   - Easy to understand by reviewers or collaborators


## 9. How to extend safely

When you want to extend this template:

- For new data sources:
  - Add modules under src/ml_tabular/data/
  - Map new configuration options into AppConfig (via config.py)
  - Write tests in tests/data/

- For new feature logic:
  - Extend tabular.py or time_series.py or create new files under src/ml_tabular/features/
  - Use configuration (YAML + Pydantic) to decide which features to apply.

- For new models:
  - Add modules under src/ml_tabular/torch/models/
  - Optionally extend training loops if you need custom behaviors.

- For new CLIs:
  - Add commands under src/ml_tabular/cli/
  - Wire them to the config system and training logic.

By keeping configuration centralized and the structure layered, you can keep adding capability without letting the project turn into a ball of mud.
