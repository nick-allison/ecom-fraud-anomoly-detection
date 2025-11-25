# <PROJECT_NAME> — Structured Deep Learning Template for Tabular & Time Series

> Note: This repository is both:
> - A concrete template project (ml-tabular-structured-template), and
> - A README template you can reuse for your own ML/DL projects.
>
> When you create a new project from this template, replace placeholders like:
> - <PROJECT_NAME>, <ONE_LINE_DESCRIPTION>, <DATASET_NAME>, etc.
> - Or rewrite sections entirely to match your actual use case.

----------------------------------------------------------------------
1. Overview
----------------------------------------------------------------------

Short pitch (replace in your own project):

<PROJECT_NAME> is a structured deep learning project focused on <ONE_LINE_DESCRIPTION> built using a reusable template for:

- Tabular and time-series problems.
- Deep learning (PyTorch) as a first-class citizen.
- Good engineering practices: config-driven runs, logging, tests, pre-commit, and experiment tracking.

In this template, you get:

- A clean Python package (ml_tabular) under src/.
- Config-driven experiments via YAML + Pydantic settings.
- Data loading layer (files, SQL, MongoDB, Kaggle).
- Feature layer for tabular and time-series engineering.
- Torch datasets/models/training loops for MLPs and basic sequence models.
- Optional MLflow integration for experiment tracking.
- A CLI for common workflows (if enabled).
- Tests, docs, and notebooks as first-class citizens.

When you copy this for a new project, you can keep the same structure and just:

- Rename the package and CLI.
- Swap in your own data, configs, and models.
- Update this README text while keeping the same sections.

----------------------------------------------------------------------
2. Project goals
----------------------------------------------------------------------

Use this section to describe what your specific project does.

For the template itself:

- Demonstrate how to structure tabular/time-series DL projects in a production-friendly way.
- Make it easy to:
  - Load data from files/DBs/Kaggle.
  - Engineer features in a repeatable, testable way.
  - Train baseline MLPs for tabular and basic models for time series.
  - Track experiments with MLflow.
  - Extend towards more complex architectures without chaos.

For your own project, replace with something like:

- Predict <TARGET> from <DATASET_NAME>.
- Provide a reproducible pipeline from raw data → features → model → metrics.
- Optionally, deploy a trained model.

----------------------------------------------------------------------
3. Features
----------------------------------------------------------------------

You can keep this list mostly as-is, and tweak it for your personal project:

- Modern Python project layout with src/ and pyproject.toml.
- Config system with Pydantic + environment profiles (dev, prod, etc.).
- Data layer:
  - CSV/Parquet/Feather loaders.
  - SQL (via SQLAlchemy).
  - MongoDB (via PyMongo, optional).
  - Kaggle integration (optional, via kaggle API).
- Feature layer:
  - Tabular: datetime expansions, log transforms, ratios, power & interaction features.
  - Time series: lags, rolling stats, windowing, date-based features.
- Deep learning layer (PyTorch):
  - Tabular MLP dataset & model.
  - Time series dataset & training loop utilities.
- Experiment tracking (MLflow):
  - Optional; toggled via config.
  - Logs params, metrics, and artifacts.
- Testing & quality:
  - pytest + coverage.
  - mypy type checking.
  - ruff linting + formatting.
  - pre-commit hooks.
- Notebooks:
  - Quickstart and EDA notebooks for tabular and time series.
- Docs (Markdown):
  - High-level overview.
  - Quickstarts.
  - Config, data, training, CLI, MLflow, dev guide.

----------------------------------------------------------------------
4. Repository structure
----------------------------------------------------------------------

Adapt this tree to match your actual project; this is the template’s structure:

    .
    ├─ pyproject.toml
    ├─ README.md
    ├─ .gitignore
    ├─ .editorconfig
    ├─ .pre-commit-config.yaml
    ├─ configs/
    │  ├─ train_tabular_baseline.yaml
    │  ├─ train_ts_baseline.yaml
    │  └─ ... (other run configs)
    ├─ src/
    │  └─ ml_tabular/
    │     ├─ __init__.py
    │     ├─ config.py
    │     ├─ exceptions.py
    │     ├─ logging_config.py
    │     ├─ data/
    │     │  ├─ __init__.py
    │     │  ├─ loading.py
    │     │  ├─ sql.py
    │     │  ├─ mongodb.py
    │     │  └─ kaggle_utils.py
    │     ├─ features/
    │     │  ├─ __init__.py
    │     │  ├─ tabular.py
    │     │  └─ time_series.py
    │     ├─ sklearn_utils/
    │     │  ├─ __init__.py
    │     │  ├─ tabular_pipeline.py
    │     │  └─ time_series_pipeline.py (optional)
    │     ├─ torch/
    │     │  ├─ __init__.py
    │     │  ├─ datasets/
    │     │  │  ├─ __init__.py
    │     │  │  ├─ tabular.py
    │     │  │  └─ time_series.py
    │     │  ├─ models/
    │     │  │  ├─ __init__.py
    │     │  │  └─ tabular_mlp.py
    │     │  └─ training/
    │     │     ├─ __init__.py
    │     │     └─ loops.py
    │     ├─ mlops/
    │     │  ├─ __init__.py
    │     │  └─ mlflow_utils.py
    │     └─ cli/
    │        ├─ __init__.py
    │        └─ app.py (Typer CLI, optional)
    ├─ train_tabular_mlp.py
    ├─ train_ts_mlp.py
    ├─ notebooks/
    │  ├─ 00_tabular_quickstart.ipynb
    │  ├─ 01_tabular_eda_and_features.ipynb
    │  ├─ 10_time_series_quickstart.ipynb
    │  ├─ 11_time_series_eda_and_windows.ipynb
    │  └─ 90_dev_scratchpad.ipynb
    ├─ docs/
    │  ├─ index.md
    │  ├─ quickstart_tabular.md
    │  ├─ quickstart_time_series.md
    │  ├─ config_and_structure.md
    │  ├─ data_and_features.md
    │  ├─ training_and_pipelines.md
    │  ├─ cli_and_scripts.md
    │  ├─ mlflow_and_experiment_tracking.md
    │  └─ contributing_and_dev.md
    └─ tests/
       ├─ conftest.py
       ├─ test_exceptions.py
       ├─ test_logging_config.py
       ├─ test_config.py
       └─ ml_tabular/
          ├─ torch/
          │  ├─ datasets/
          │  │  ├─ test_tabular_dataset.py
          │  │  └─ test_time_series_dataset.py
          │  ├─ models/
          │  │  └─ test_tabular_mlp.py
          │  └─ training/
          │     └─ test_training_loops.py
          ├─ cli/
          │  └─ test_cli.py
          └─ mlops/
             └─ test_mlflow_utils.py

For your own project, modify filenames (e.g. train_churn_mlp.py) but keep the conceptual layout.

----------------------------------------------------------------------
5. Getting started
----------------------------------------------------------------------

5.1 Prerequisites

- Python >= 3.10, < 3.13.
- Git.
- Recommended: virtualenv, conda, or uv for environment management.

5.2 Installation

Clone the repo and install in editable mode:

    git clone https://github.com/<YOUR_USERNAME>/<YOUR_REPO>.git
    cd <YOUR_REPO>

    python -m venv .venv
    source .venv/bin/activate   # on Windows: .venv\Scripts\activate

    pip install -e ".[dev,mlops,validation]"

This installs:

- Core dependencies (numpy, pandas, scikit-learn, torch, etc.).
- Dev tools (pytest, mypy, ruff, pre-commit, jupyter).
- Optional MLflow + Kaggle and Pandera (if you keep those extras).

5.3 Pre-commit hooks (recommended)

Enable pre-commit so style and basic checks run before each commit:

    pre-commit install
    pre-commit run --all-files

----------------------------------------------------------------------
6. Configuration
----------------------------------------------------------------------

The project uses ml_tabular.config.AppConfig + YAML files:

- Default config filenames:
  - config.yaml or config.yml at the repo root (or as set by ML_TABULAR_CONFIG_PATH).
- Environment profiles:
  - dev, prod, test etc. selected via ML_TABULAR_ENV.

Examples:

- Environment variables:
  - ML_TABULAR_ENV=dev
  - ML_TABULAR_CONFIG_PATH=./configs/config.dev.yaml

Config fields include:

- paths: base, raw, processed, models directories.
- training: generic hyperparameters.
- database and mongo: for SQL/Mongo connections (optional).
- mlflow: enable/disable, tracking URI, experiment name.
- kaggle: default dataset/competition, download directory.
- time_series: global hints (datetime column, horizon, window length, etc.).

See docs/config_and_structure.md for a deeper dive, and look at
configs/train_tabular_baseline.yaml and configs/train_ts_baseline.yaml for per-run
settings.

----------------------------------------------------------------------
7. Data & features
----------------------------------------------------------------------

7.1 Data loading

Use these modules:

- ml_tabular.data.loading:
  - load_dataframe(path, ...)
  - load_raw_dataset(filename, ...)
  - load_processed_dataset(filename, ...)
- ml_tabular.data.sql:
  - get_engine(), load_sql_query(...), load_sql_table(...), execute_sql(...)
- ml_tabular.data.mongodb:
  - get_collection(...), load_mongo_collection(...), count_documents(...)
- ml_tabular.data.kaggle_utils:
  - Helpers to download datasets/competitions from Kaggle to data/raw/kaggle/.

Under the hood, errors are raised with DataError (subclass of AppError) with
rich context for debugging.

7.2 Feature engineering

Tabular features (ml_tabular.features.tabular):

- Define a FeatureSpec with:
  - datetime_columns
  - log1p_columns
  - ratio_features (num/den)
  - power_features (x^p)
  - interaction_features (x*y)
- Apply build_features(df, spec=...) to create new columns and optionally
  drop originals.

Time-series features (ml_tabular.features.time_series):

- Add or refine:
  - Lag features.
  - Rolling statistics.
  - Time-based features (day of week, hour, etc.).
- Combine with your dataset’s windowing logic in ml_tabular.torch.datasets.time_series.

In a real project, you’d describe exactly what features you use and why they
matter for your problem.

----------------------------------------------------------------------
8. Models & training
----------------------------------------------------------------------

8.1 Tabular training (MLP)

Script: train_tabular_mlp.py

- Reads a config (baseline or custom).
- Loads raw/processed data.
- Applies tabular feature engineering.
- Builds a TabularDataset and wraps it in DataLoaders.
- Constructs a TabularMLP model (ml_tabular.torch.models.tabular_mlp).
- Trains with helper functions from ml_tabular.torch.training.loops.

Basic usage (example):

    python train_tabular_mlp.py --config configs/train_tabular_baseline.yaml

Or, if you have a CLI exposed (via Typer):

    ml-tabular train-tabular --config configs/train_tabular_baseline.yaml

8.2 Time-series training

Script: train_ts_mlp.py

- Similar structure, but uses:
  - Time-series specific configs (train_ts_baseline.yaml).
  - Time-series features and windowing.
  - TimeSeriesDataset for sequence inputs & targets.

Example:

    python train_ts_mlp.py --config configs/train_ts_baseline.yaml

8.3 Extending models

For your own project, describe your actual models here, for example:

- TabularMLP: layers, activations, regularization, etc.
- Time-series model: LSTM/TCN/Transformer, horizon, loss function.

Summarize key hyperparameters and where they are configured in YAML.

----------------------------------------------------------------------
9. Experiment tracking (MLflow)
----------------------------------------------------------------------

If you install the mlops extra and enable MLflow in config:

- ml_tabular.mlops.mlflow_utils provides helper functions to:
  - Start runs.
  - Log parameters/metrics.
  - Log artifacts (models, configs, plots).

Typical pattern inside a training script:

1. Load config.
2. If cfg.mlflow.enabled:
   - Initialize MLflow using mlflow_utils (set tracking URI, experiment).
3. Wrap training loop in an MLflow run:
   - Log hyperparameters and metrics at each epoch.
   - Log final model and config.

In this section, for your own project, link to the actual experiments and maybe show:

- Example MLflow run screenshot (in your repo screenshot folder).
- How to reproduce the top-performing run.

----------------------------------------------------------------------
10. CLI usage
----------------------------------------------------------------------

If ml_tabular.cli is present and wired to [project.scripts] in pyproject.toml:

- Entry point: ml-tabular

Example commands (adapt to your actual CLI):

    # List available commands
    ml-tabular --help

    # Train tabular baseline
    ml-tabular train-tabular --config configs/train_tabular_baseline.yaml

    # Train time-series baseline
    ml-tabular train-ts --config configs/train_ts_baseline.yaml

    # (Optional) Run quick EDA or diagnostics commands, if you add them
    ml-tabular inspect-config --config configs/train_tabular_baseline.yaml

For your specific project, document each command and the most important flags.

----------------------------------------------------------------------
11. Development workflow
----------------------------------------------------------------------

11.1 Running tests

Run the full test suite:

    pytest

Run fast unit tests only (example, if you tag slow tests):

    pytest -m "not slow"

Run a specific test file:

    pytest tests/ml_tabular/torch/models/test_tabular_mlp.py

11.2 Linting & formatting

Use ruff directly:

    ruff check src tests
    ruff format src tests

Or rely on pre-commit (recommended):

    pre-commit run --all-files

11.3 Type checking

    mypy src

For your own project, you can add a short “expected” runtime (e.g. “Tests take ~X seconds”)
and notes on CI if you configure GitHub Actions.

----------------------------------------------------------------------
12. Using this template for your own project
----------------------------------------------------------------------

Checklist for turning this into a new project:

1. Create a new repo:
   - Use this template repo as a starting point (or clone and push to a new repo).

2. Rename the package:
   - Replace ml_tabular with something like churn_tabular or my_project.
   - Update:
     - src/ml_tabular/ → src/<your_package>/
     - Imports in all modules.
     - pyproject.toml:
       - [tool.mypy].packages
       - [tool.pytest.ini_options].addopts coverage target
       - [tool.setuptools.packages.find].include
       - [project.scripts] entry point.

3. Update README placeholders:
   - <PROJECT_NAME>, <ONE_LINE_DESCRIPTION>, dataset names, etc.
   - Rewrite “Project goals” and “Models & training” sections to match your problem.

4. Set up configs & data:
   - Create dataset-specific configs in configs/.
   - Put example data or describe how to obtain it in this README.
   - Define appropriate FeatureSpec for your dataset.

5. Tune training scripts:
   - Adjust hyperparameters and architectures in:
     - train_tabular_mlp.py
     - train_ts_mlp.py
   - Or add new training scripts (e.g. train_churn_mlp.py, train_traffic_forecast.py).

6. Wire MLflow / Kaggle if needed:
   - Set up environment variables and config entries.
   - Test MLflow logging and Kaggle downloads.

7. Keep docs and notebooks in sync:
   - Update docs/*.md and notebooks/* to reflect your actual problem and data.
   - Use notebooks as a narrative companion to this README.

Once you’ve done this once or twice, you’ll have:

- A consistent pattern for all your ML/DL projects.
- A portfolio that looks and feels cohesive and professional.

----------------------------------------------------------------------
13. Roadmap
----------------------------------------------------------------------

For the template itself, future enhancements might include:

- More model architectures (e.g., gradient boosting wrappers, time-series transformers).
- Additional preprocessing pipelines (categorical embeddings, advanced normalization).
- Example integration with deployment frameworks (FastAPI, Streamlit, etc.).
- More opinionated schema validation using Pandera.

For your own project, use this section to outline planned features and TODOs.

----------------------------------------------------------------------
14. License
----------------------------------------------------------------------

Specify the license for your project here. For this template, a common choice is:

- MIT License (see LICENSE file).

For your derived projects, you can keep MIT or choose something else depending on your needs.
