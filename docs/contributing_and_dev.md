# Contributing and Development Guide

This document explains how to work on the `ml_tabular` template as a developer:

- How to set up your environment.
- How to run tests and checks (pytest, mypy, ruff, pre-commit).
- How to work with configs, data, and notebooks.
- How to propose changes, structure branches, and keep things clean.

Even if you’re the main (or only) contributor, following these practices makes your work more reliable and easier to extend later.


## 1. Repository structure (developer view)

At a high level, the project looks like:

- `pyproject.toml`
  - Project metadata, dependencies, and tool configs (ruff, mypy, pytest).
- `.pre-commit-config.yaml`
  - Pre-commit hooks (ruff, mypy, basic hygiene).
- `.editorconfig`
  - Editor/IDE formatting defaults.
- `.gitignore`
  - Ignore patterns for Python, data, models, logs, etc.
- `src/ml_tabular/`
  - Main Python package.
  - Submodules:
    - `config.py` — app-wide configuration.
    - `exceptions.py` — structured error types.
    - `logging_config.py` — logging setup.
    - `data/` — loading from files, SQL, MongoDB, Kaggle, etc.
    - `features/` — tabular + time-series feature engineering.
    - `sklearn_utils/` — preprocessing pipelines (tabular/time-series).
    - `torch/` — datasets, models, and training loops for deep learning.
    - `mlops/` — MLflow integration, experiment tracking helpers.
    - `cli/` — Typer-based command-line interface (optional, if present).
- `configs/`
  - YAML run configs (e.g., `train_tabular_baseline.yaml`, `train_ts_baseline.yaml`).
- `notebooks/`
  - Example notebooks for EDA, quickstarts, and dev scratchpads.
- `docs/`
  - Markdown docs (these files) that explain how to use the template.
- `tests/`
  - Unit tests organized by module (exceptions, config, datasets, models, training, cli, mlops, etc.).
- `data/`, `models/`, `logs/`
  - Created at runtime; usually not committed to git.


## 2. Development environment setup

### 2.1 Clone and create a virtual environment

Basic steps:

1) Clone the repository:
   - `git clone <repo-url>`
   - `cd ml-tabular-structured-template`

2) Create a virtual environment (examples):

   - Python venv:
     - `python -m venv .venv`
     - `source .venv/bin/activate`  (Linux/macOS)
     - `.venv\Scripts\activate`     (Windows)

   - Or use your preferred tool (conda, uv, etc.), as long as it matches:
     - `python >= 3.10, < 3.13` (as specified in `pyproject.toml`).

3) Install the package in editable mode with dev extras:

   - `pip install -e ".[dev]"`

   - Optionally add more extras:
     - `pip install -e ".[dev,mlops,validation]"`


### 2.2 Optional extras

- `mlops` extra:
  - Installs:
    - `mlflow`
    - `kaggle`
  - Required for MLflow tracking and Kaggle integration features.

- `validation` extra:
  - Installs:
    - `pandera` (for DataFrame schema validation, if you add it later).

- You can always add more extras to the `pyproject.toml` as the template grows.


## 3. Code style and formatting

### 3.1 Ruff (lint + format)

The project uses `ruff` for:

- Linting:
  - Checks for:
    - basic errors (E/F),
    - bug patterns (B),
    - import sorting (I),
    - modernization (UP),
    - simplify (SIM),
    - logging formatting (G).
- Formatting:
  - `ruff format` is used as the code formatter.
  - Line length is configured (e.g. 100 characters).
  - Docstring code formatting enabled.

Key points:

- You should **not** manually run `isort` or another formatter if `ruff format` is configured.
- `pre-commit` will run ruff and ruff-format before commits.


### 3.2 Type checking (mypy)

The project uses `mypy` with settings like:

- `python_version = "3.10"`
- `packages = ["ml_tabular"]`
- `ignore_missing_imports = true` (for third-party libs).
- `disallow_untyped_defs = true`, `check_untyped_defs = true`.
- `strict_optional = true`.

Implications:

- Public functions and classes should be fully typed.
- New code should include type hints for:
  - Parameters.
  - Return values.
  - Attributes.

This gives you:

- Early detection of incompatible refactors.
- Stronger guarantees when building pipelines and wrappers.


### 3.3 EditorConfig

`.editorconfig` defines:

- Spaces vs tabs.
- Indentation sizes.
- Line endings (`lf`).
- Trailing whitespace trimming (except for Markdown).

Make sure your editor/IDE respects these settings. Most modern editors do so automatically if the EditorConfig extension is enabled.


## 4. Pre-commit hooks

### 4.1 Installing and enabling pre-commit

Once you have the environment set up:

1) Install `pre-commit` (already included in `[project.optional-dependencies].dev`).
2) Run:
   - `pre-commit install`

This ensures:

- Hooks run automatically on each `git commit`:
  - Ruff lint & format.
  - Mypy (optional; can be slow).
  - Basic hygiene checks:
    - `check-yaml`
    - `end-of-file-fixer`
    - `trailing-whitespace`
    - `check-merge-conflict`
    - `check-added-large-files`

You also have an optional local hook:

- `pytest` (quick tests) configured with:
  - `stages: [manual]`
- This means:
  - It doesn’t run automatically on every commit.
  - You can run:
    - `pre-commit run pytest` to quickly exercise tests.


### 4.2 Running hooks manually

At any time you can run:

- `pre-commit run --all-files`

This applies the hooks to the entire repo and is a good sanity check before:

- Opening a PR.
- Pushing significant changes.


## 5. Running tests

### 5.1 Pytest basics

The project uses `pytest` with configuration in `pyproject.toml`:

- `testpaths = ["tests"]`
- `addopts = "-ra --cov=ml_tabular --cov-report=term-missing"`
- Custom markers:
  - `slow` (long-running tests).
  - `integration` (tests hitting external services or multiple components).

To run the test suite:

- `pytest`

To skip slow tests:

- `pytest -m "not slow"`

To run only integration tests:

- `pytest -m integration`

To run a specific test module or function:

- `pytest tests/test_config.py`
- `pytest tests/ml_tabular/torch/models/test_tabular_mlp.py::test_forward_pass_shapes`


### 5.2 Test layout

Typical structure:

- `tests/test_exceptions.py`
  - Ensures:
    - `AppError` and derived exceptions behave as expected.
    - `to_dict` produces a good JSON-serializable shape.

- `tests/test_logging_config.py`
  - Verifies:
    - Logging configuration doesn’t crash.
    - Loggers can be obtained.
    - Basic log messages are emitted.

- `tests/test_config.py`
  - Confirms:
    - `AppConfig` loads correctly from dict/YAML.
    - Profiles (dev/prod) are correctly selected.
    - Validation is enforced.

- `tests/ml_tabular/torch/datasets/test_tabular_dataset.py`
  - Tests:
    - `TabularDataset` indexing.
    - Shapes of samples.
    - Behavior when optional features/targets are absent.

- `tests/ml_tabular/torch/datasets/test_time_series_dataset.py`
  - Tests:
    - `TimeSeriesDataset` slicing.
    - Window creation logic.
    - Grouped series behavior (if implemented).

- `tests/ml_tabular/torch/models/test_tabular_mlp.py`
  - Tests:
    - MLP forward pass.
    - Behavior with different input dims.
    - Basic gradient/backprop flow (optionally).

- `tests/ml_tabular/torch/training/test_training_loops.py`
  - Tests:
    - Single-epoch training loop on a tiny synthetic dataset.
    - No crash when metrics are logged or MLflow is enabled/disabled (mocked).

- `tests/ml_tabular/cli/test_cli.py`
  - Tests:
    - CLI command registration.
    - Argument parsing and top-level options.
    - Dry-run or smoke tests for training entrypoints.

- `tests/ml_tabular/mlops/test_mlflow_utils.py`
  - Tests:
    - MLflow helper functions.
    - Behavior when MLflow is not installed (if optional).
    - Behavior with `mlflow.enabled = False` vs `True` (mocked).


## 6. Configuration and secrets in development

### 6.1 `config.yaml` and profiles

You can keep a `config.yaml` at the project root, with optional profiles:

- Example:

  - `dev:` section
  - `prod:` section

The code uses:

- `ML_TABULAR_ENV` to pick the active profile.
- `ML_TABULAR_CONFIG_PATH` to override the config file location if needed.

For local development:

- You might have:
  - `ML_TABULAR_ENV=dev`
  - `config.yaml` with:
    - Paths pointing to local data.
    - `mlflow.enabled = true` or `false` as you prefer.
    - DB credentials using environment variables only (not hard-coded in YAML).


### 6.2 Sensitive information

**Never** commit:

- API keys (Kaggle, MLflow, DB credentials).
- `.env` files with secrets.

Instead:

- Use a local `.env` file (which is gitignored) with:
  - `MLFLOW_TRACKING_URI=...`
  - `KAGGLE_USERNAME=...`
  - `KAGGLE_KEY=...`
  - DB password env vars.

The config layer is designed to read from:

- YAML.
- Environment variables.
- `.env` via `pydantic_settings.BaseSettings`.


## 7. Working with data, features, and models in dev

### 7.1 Data layer (dev workflow)

For local experimentation, you can:

- Put raw data into `data/raw/`:
  - e.g. `data/raw/titanic.csv`.
- Use:
  - `ml_tabular.data.loading.load_raw_dataset(...)`
  - `ml_tabular.data.sql.load_sql_query(...)`
  - `ml_tabular.data.mongodb.load_mongo_collection(...)` (if Mongo is used).
- Validate shapes and required columns using:
  - `DataError` when things go wrong.
  - Optional `validation` extras like `pandera` (if you extend the template).

This separates **data ingestion** from **feature engineering** and **modeling**, making it easier to debug and reuse.


### 7.2 Feature layer

For tabular data:

- Use `ml_tabular.features.tabular.FeatureSpec` to define:
  - Datetime expansions.
  - `log1p` features.
  - Ratio features.
  - Power and interaction features.

You can:

- Keep a default `FeatureSpec` in `tabular.py`.
- Override or specialize it per project or per dataset.

For time series:

- Use `ml_tabular.features.time_series` to:
  - Add lag features.
  - Rolling statistics.
  - Date/time-based expansions (day of week, etc.), if appropriate.

The goal is:

- Stateless, pure functions for feature engineering.
- No training state in feature code (no scalers/encoders here).
- Training-time state (like scalers) belongs in preprocessing pipelines or training scripts.


### 7.3 Model and training layer

Torch-based models live in:

- `ml_tabular.torch.models`
  - Example: `TabularMLP`.

Datasets live in:

- `ml_tabular.torch.datasets`
  - Example: `TabularDataset`, `TimeSeriesDataset`.

Training loops live in:

- `ml_tabular.torch.training.loops`
  - Example: `train_one_epoch`, `evaluate`, `fit_model`.

The pattern:

1) Load config and data.
2) Create features (stateless transforms).
3) Build dataset and DataLoader.
4) Initialize model + optimizer + scheduler.
5) Call training loop utilities.
6) Log metrics and artifacts (optionally via MLflow).


## 8. CLI and scripts

If `ml_tabular.cli` is present, there will be:

- A Typer-based CLI app.
- `pyproject.toml` exposes a console script:
  - e.g. `ml-tabular = "ml_tabular.cli:app"`.

Usage examples:

- `ml-tabular train-tabular --config configs/train_tabular_baseline.yaml`
- `ml-tabular train-ts --config configs/train_ts_baseline.yaml`

As a developer:

- Keep CLI commands thin:
  - Parse arguments.
  - Call well-factored Python functions (from `train_tabular_mlp.py` / `train_ts_mlp.py`).
- This ensures:
  - CLI is easy to use for quick experiments.
  - Core logic remains testable (pure Python functions, no CLI glue inside tests).


## 9. MLflow in development

If MLflow is enabled (via config + extras):

- Use it for experiment tracking even during development:
  - Helps you compare changes across branches and refactors.
- But keep it flexible:
  - If you’re doing quick prototyping and don’t want tracking:
    - Set `mlflow.enabled = False` in your dev config.
    - Or pass `--no-mlflow` flags if you add them to the CLI.

MLflow integration is centralized in `ml_tabular.mlops.mlflow_utils`, so:

- Any future improvements (tags, extra logging, registry integration) happen there.
- Training code just uses the helpers.


## 10. Branching, PRs, and coding guidelines

### 10.1 Branching strategy

Even for solo work, using simple branching helps:

- `main` or `master`:
  - Always in a working state.
  - Tagged releases for “template versions”.
- Feature branches:
  - `feature/add-time-series-lstm`
  - `feature/improve-mlflow-logging`
  - `fix/sql-connection-errors`

Typical workflow:

1) Branch from `main`.
2) Make changes.
3) Run:
   - `pre-commit run --all-files`
   - `pytest`
4) Merge back via PR (even if self-reviewed).


### 10.2 Coding style

- Prefer small, focused modules and functions.
- Keep side effects (I/O, network calls, DB) near edges:
  - CLI.
  - Data loading modules.
  - Training scripts.
- Keep core logic pure where possible:
  - Feature transformations.
  - Dataset indexing.
  - Loss calculations.

- Use typed exceptions (`ConfigError`, `DataError`, `ModelError`, `PipelineError`) instead of:
  - Returning error codes.
  - Bare `Exception`.
- Log context-rich messages:
  - Use `extra={...}` in logging calls to include:
    - Paths.
    - Dataset names.
    - Column names.
    - Hyperparameters.


## 11. Extending the template

As you build more projects on top of this template, you may:

1) Add more feature engineering utilities:
   - Domain-specific feature sets.
   - Target encodings (carefully applied, with train/test separation).

2) Add more model architectures:
   - Wider/deeper MLPs.
   - Time-series models (e.g. LSTMs, temporal convolution, transformers).
   - Hybrid/tabular + categorical embeddings.

3) Add more MLOps integrations:
   - Model registry (via MLflow).
   - Deployment scripts (FastAPI endpoints).
   - Batch scoring pipelines.

4) Add more tests:
   - Property-based tests (e.g. via `hypothesis`) for data transformations.
   - Performance regression checks on synthetic data.

The current structure is designed to scale with these additions without becoming tangled.


## 12. Summary

As a contributor (even if the “team” is just you), the development workflow is:

1) Set up environment:
   - `pip install -e ".[dev,mlops,validation]"` (as needed).
2) Enable pre-commit:
   - `pre-commit install`
3) Make changes in:
   - `src/ml_tabular/...`
   - `configs/...`
   - `docs/...`
   - `notebooks/...`
4) Keep everything passing:
   - `pre-commit run --all-files`
   - `pytest`
5) Use MLflow where helpful:
   - `cfg.mlflow.enabled = true` for tracked experiments.
6) Keep code:
   - Typed.
   - Logged.
   - Config-driven.

This gives you a robust, professional-grade ML template that showcases:

- Good software engineering practices.
- Deep-learning readiness (Torch).
- Data and experiment discipline (config, logging, MLflow).
