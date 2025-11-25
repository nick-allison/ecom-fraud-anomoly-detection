# ML Tabular Structured Template

A batteries-included, opinionated template for tabular and time-series deep-learning projects using PyTorch, scikit-learn, and modern Python tooling.

This repo is designed to show that you can walk into a new environment and immediately bring:

- Clear, maintainable project structure
- Strong engineering discipline (tests, configs, logging, MLOps hooks)
- Practical, production-aware ML practices that work in real teams

It is intentionally small enough to understand in one sitting, but structured enough to scale to serious projects.


## 1. What this template is for

This template focuses on:

1) Supervised learning on tabular data
   - Classification and regression
   - Feature engineering for numeric and categorical columns
   - Baseline models and deep learning models

2) Time-series forecasting / sequence models
   - Single or multiple related series (store, sensor, device, etc.)
   - Time windowing and lag/rolling features
   - PyTorch datasets for sequence models (MLP or sequence backbones)

3) Modern Python project hygiene
   - pyproject.toml with pinned tooling configuration
   - ruff, mypy, pytest, pre-commit
   - Centralized logging and structured exceptions
   - Config management with pydantic-settings

Optional add-ons:

- MLflow integration (metrics + artifacts)
- Kaggle integration for downloading datasets


## 2. High-level architecture

At a high level, there are three layers:

1) Infrastructure layer (shared plumbing)
   - Config, logging, error handling
   - Data loading from files / SQL / MongoDB / Kaggle
   - MLflow + experiment tracking hooks

2) Domain layer (tabular + time-series logic)
   - Tabular feature building
   - Time-series feature building (lags, diffs, rolling statistics)
   - Torch datasets, models, and training loops

3) Interface layer (entrypoints)
   - CLI (ml_tabular) to launch common workflows
   - Top-level training scripts:
     - train_tabular_mlp.py
     - train_ts_mlp.py
   - Notebooks for EDA and experiments

This separation lets you:

- Reuse infrastructure across multiple projects
- Plug in different models or experiments
- Keep your “public API” (CLI + scripts + notebooks) clean and focused


## 3. Project layout

A typical layout for this template:

(Use this as a reference; exact structure may vary slightly.)

.
├── pyproject.toml
├── README.md
├── LICENSE
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
├── docs/
│   └── index.md
├── configs/
│   ├── train_tabular_baseline.yaml
│   └── train_ts_baseline.yaml
├── notebooks/
│   ├── 00_tabular_quickstart.ipynb
│   ├── 01_tabular_eda_and_features.ipynb
│   ├── 10_time_series_quickstart.ipynb
│   ├── 11_time_series_eda_and_windows.ipynb
│   └── 90_dev_scratchpad.ipynb
├── src/
│   └── ml_tabular/
│       ├── __init__.py
│       ├── config.py
│       ├── exceptions.py
│       ├── logging_config.py
│       ├── data/
│       │   ├── __init__.py
│       │   ├── loading.py
│       │   ├── sql.py
│       │   ├── mongodb.py
│       │   └── kaggle.py
│       ├── features/
│       │   ├── __init__.py
│       │   ├── tabular.py
│       │   └── time_series.py
│       ├── sklearn_utils/
│       │   ├── __init__.py
│       │   └── tabular_pipeline.py
│       ├── torch/
│       │   ├── __init__.py
│       │   ├── datasets/
│       │   │   ├── __init__.py
│       │   │   ├── tabular.py
│       │   │   └── time_series.py
│       │   ├── models/
│       │   │   ├── __init__.py
│       │   │   └── tabular_mlp.py
│       │   └── training/
│       │       ├── __init__.py
│       │       └── loops.py
│       ├── mlops/
│       │   ├── __init__.py
│       │   └── mlflow_utils.py
│       └── cli/
│           ├── __init__.py
│           └── app.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_exceptions.py
    ├── test_logging_config.py
    ├── test_config.py
    ├── torch/
    │   ├── __init__.py
    │   ├── datasets/
    │   │   ├── __init__.py
    │   │   ├── test_tabular_dataset.py
    │   │   └── test_time_series_dataset.py
    │   ├── models/
    │   │   ├── __init__.py
    │   │   └── test_tabular_mlp.py
    │   └── training/
    │       ├── __init__.py
    │       └── test_training_loops.py
    ├── cli/
    │   ├── __init__.py
    │   └── test_cli.py
    └── mlops/
        ├── __init__.py
        └── test_mlflow_utils.py


## 4. Core building blocks

4.1 Configuration (ml_tabular.config)

Centralized configuration via pydantic-settings:

- Environment-aware:
  - Uses ML_TABULAR_ENV and ML_TABULAR_CONFIG_PATH
  - Supports profile-based YAML (for example: dev, prod sections)
- Structured config sections:
  - paths: data directories, models directory
  - training: generic training parameters
  - database: SQLAlchemy-style DB config
  - mongo: MongoDB settings
  - mlflow: experiment tracking options
  - kaggle: default dataset and competition identifiers
  - time_series: TS-specific hints (window sizes, horizon, etc.)

Example usage:

from ml_tabular.config import get_config

cfg = get_config()
paths = cfg.resolved_paths()
print(paths.raw_dir)

This gives your scripts access to consistent, validated configuration with overrides via:

- YAML configs under configs/
- Environment variables
- Optional .env file in development


4.2 Logging (ml_tabular.logging_config)

Central logging configuration that is environment-aware:

- Console logging in dev by default
- Optional rotating file handler in production-like environments
- Environment variables:
  - LOG_LEVEL
  - LOG_DIR
  - ML_TABULAR_ENV
  - ML_TABULAR_LOG_FORMAT (text or json)

Example:

from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)
logger.info("Starting training", extra={"script": "train_tabular_mlp"})


4.3 Exceptions (ml_tabular.exceptions)

Structured exception hierarchy:

- AppError base class with:
  - code, message, cause, context, location
  - to_dict() for logging or APIs
- Domain-specific subclasses:
  - ConfigError
  - DataError
  - ModelError
  - PipelineError

This makes error logging and debugging much clearer and more consistent.


4.4 Data loading (ml_tabular.data)

Unified loading layer that abstracts away I and O concerns:

- loading.py
  - load_dataframe(path, format, required_columns, read_kwargs)
  - load_raw_dataset(filename, ...)
  - load_processed_dataset(filename, ...)
- sql.py
  - get_engine() from config
  - load_sql_query(), load_sql_table(), execute_sql()
- mongodb.py
  - get_mongo_client(), get_collection()
  - load_mongo_collection(), count_documents()
- kaggle.py
  - Helpers to download Kaggle datasets or competitions into data/raw/kaggle/

All raw data entry points are here so the rest of the code can assume it is working with clean DataFrames and not worry about credentials or drivers.


4.5 Feature engineering (ml_tabular.features)

Two modules:

- tabular.py
  - FeatureSpec describing:
    - datetime_columns
    - log1p_columns
    - ratio_features
    - power_features
    - interaction_features
  - build_features(df, spec):
    - Rich datetime decomposition
    - Log1p transforms for non-negative numeric data
    - Ratio, power, and interaction features

- time_series.py
  - TimeSeriesFeatureSpec describing:
    - datetime_column, group_column
    - sort_by_time, expand_datetime
    - lag_features, diff_features, rolling_features
  - build_time_series_features(df, spec):
    - Sorting and grouping by time
    - Lag and diff features
    - Rolling window statistics

These specs are declarative so you can:

- Version-control your feature choices
- Reuse the same transformations across train, validation, and test
- Make the feature logic explicit and testable


4.6 PyTorch datasets and models (ml_tabular.torch)

Datasets:

- datasets/tabular.py
  - TabularDataset for generic tabular problems
    - Accepts DataFrames and column lists
    - Returns (x, y) as torch.Tensors

- datasets/time_series.py
  - TimeSeriesWindowDataset
    - Windowed sequences for forecasting
    - Supports multiple series via a group_column
    - Handles input windows and horizons

Models:

- models/tabular_mlp.py
  - Flexible MLP for tabular inputs:
    - Configurable hidden layers, activation, dropout
    - Works for regression or classification
  - Intended as a baseline or template you can swap out later

Training loops:

- training/loops.py
  - Reusable training utilities:
    - train_epoch, evaluate
    - Metric aggregation
    - Optional early stopping or checkpoint hooks depending on your implementation


4.7 MLOps utilities (ml_tabular.mlops)

- mlflow_utils.py
  - Thin wrappers around MLflow:
    - start_run, log_params, log_metrics, log_artifacts
  - Respects MlflowConfig in config.py
  - Lets you:
    - Turn MLflow on or off per environment
    - Standardize what gets logged for each run


4.8 CLI (ml_tabular.cli.app)

A typer-based CLI wrapping common workflows.

Examples (exact command names depend on implementation):

- ml-tabular train-tabular --config configs/train_tabular_baseline.yaml
- ml-tabular train-ts --config configs/train_ts_baseline.yaml

This gives you:

- A “single command” UX for common tasks
- A central place for argument parsing and orchestration
- A clean story to tell: “All standard tasks are exposed via one CLI.”


## 5. Training entrypoints

At the repository root:

1) train_tabular_mlp.py
   - Reads configs/train_tabular_baseline.yaml
   - Loads data via ml_tabular.data.loading
   - Applies tabular features
   - Builds a TabularDataset
   - Instantiates TabularMLP
   - Trains using training.loops
   - Optionally logs to MLflow

2) train_ts_mlp.py
   - Reads configs/train_ts_baseline.yaml
   - Loads time-series data
   - Applies time-series features (lags, windows)
   - Builds a TimeSeriesWindowDataset
   - Trains a forecasting model


## 6. Configuration files in configs/

Examples:

- train_tabular_baseline.yaml
  - Paths to raw or processed datasets
  - Columns for features and targets
  - FeatureSpec-like settings (datetime, ratios, interactions)
  - MLP hyperparameters
  - Training settings: batch size, epochs, learning rate, etc.

- train_ts_baseline.yaml
  - Time-series fields:
    - datetime_column, group_column
    - input_window, prediction_horizon
  - Time-series feature choices: lags, diffs, rolling stats
  - Model and training hyperparameters

The goal is that your experiment is fully described by:

- The YAML config
- The git commit of this template


## 7. Notebooks

The notebooks directory is designed to be both didactic and practical.

Suggested notebooks:

- 00_tabular_quickstart.ipynb
  - Small end-to-end run on a toy dataset using the template
- 01_tabular_eda_and_features.ipynb
  - EDA and feature exploration, and how it maps into FeatureSpec
- 10_time_series_quickstart.ipynb
  - Basic time-series forecasting walkthrough
- 11_time_series_eda_and_windows.ipynb
  - Visualizing windows, lags, and rolling features
- 90_dev_scratchpad.ipynb
  - A free-form playground for experiments and rough work


## 8. Tests

The tests directory demonstrates:

- Unit tests for core infrastructure:
  - Exceptions
  - Logging configuration
  - Config loading and validation
- Tests for PyTorch components:
  - Dataset shapes and data types
  - Simple forward pass through TabularMLP
  - Basic training-loop sanity (single batch, single step)
- CLI smoke tests:
  - Verifying CLI commands parse and run without crashing
- MLflow helper tests (mocked):
  - Ensuring that enabling MLflow does not break training scripts

This shows:

- You know how to test ML infrastructure
- You care about correctness and maintainability, not just model accuracy


## 9. How to use this template

9.1 New project based on this template

1) Create a new repository using this layout.
2) Update metadata:
   - pyproject.toml
   - README.md
3) Define configuration:
   - Copy configs/train_tabular_baseline.yaml and adapt it
   - Or create configs/train_ts_baseline.yaml for time-series
4) Make data available:
   - Put files in data/raw, or
   - Wire up SQL or MongoDB, or
   - Use the Kaggle utilities
5) Run a baseline:
   - python train_tabular_mlp.py --config configs/train_tabular_baseline.yaml
   - or use the CLI if configured
6) Iterate:
   - Change YAML configs for most hyperparameters
   - Enhance features in features/tabular.py or features/time_series.py
   - Track runs in MLflow if enabled

9.2 Adapting to a new employer or domain

You can:

- Fork this structure (mentally or literally)
- Tailor:
  - config.py to their infra and environments
  - logging_config.py to their logging platform
  - mlops/mlflow_utils.py or equivalents to their tracking stack
- Plug in their data sources and domain-specific logic

This demonstrates that you can bring order and discipline to ML projects while keeping enough flexibility for research and iteration.


## 10. Extension ideas

Some natural extensions:

- Additional model types:
  - Gradient boosting (xgboost, lightgbm)
  - More complex deep nets (transformers, sequence models)
- Infra:
  - Dockerfile and simple containerization story
  - CI workflows for tests and linting
- Documentation:
  - Example projects using this template
  - Benchmarks for standard datasets
