# Training and Pipelines

This document explains how **training** and **pipelines** are structured in the `ml_tabular` template, for both **tabular** and **time-series** problems.

The main goals are:

- Make training scripts **predictable and reproducible**.
- Keep most logic in **reusable library modules** rather than ad-hoc notebooks.
- Provide a clean path to:
  - Baseline models (sklearn-style).
  - Deep learning models (PyTorch).
  - Config-driven experiment management.
  - Optional MLflow tracking for experiments.


## 1. High-level architecture

At a high level, training in this template follows a consistent pattern:

1) Configuration
   - A YAML file (e.g. `configs/train_tabular_baseline.yaml`, `configs/train_ts_baseline.yaml`)
   - Global configuration via `ml_tabular.config.AppConfig` (paths, DB, MLflow, Kaggle, time-series hints).

2) Data and features
   - Data loaded via `ml_tabular.data.*` modules (CSV, Parquet, SQL, MongoDB, Kaggle).
   - Feature engineering via:
     - `ml_tabular.features.tabular` for general tabular features.
     - `ml_tabular.features.time_series` for time-based features.

3) Datasets and pipelines
   - For traditional tabular ML:
     - Scikit-learn pipeline built via `ml_tabular.sklearn_utils.tabular_pipeline.TabularPipelineBuilder`.
   - For deep learning:
     - PyTorch datasets (`TabularDataset`, `TimeSeriesDataset`) with DataLoader(s).

4) Training loop
   - Centralized in `ml_tabular.torch.training.loops`:
     - Handles epochs, optimization, gradient clipping, early stopping, etc.

5) Experiment tracking (optional)
   - MLflow utilities in `ml_tabular.mlops.mlflow_utils`.

6) Entry points
   - Root-level scripts:
     - `train_tabular_mlp.py`
     - `train_ts_mlp.py`
   - CLI commands via `ml_tabular.cli` (using Typer).


## 2. Configuration-driven training

Two main layers of configuration work together:

1) **Global application config** (`AppConfig` from `ml_tabular.config`)

   - Defines:
     - Paths:
       - `paths.base_dir`
       - `paths.data_dir`
       - `paths.raw_dir`
       - `paths.processed_dir`
       - `paths.models_dir`
     - Training defaults:
       - `training.random_seed`
       - `training.test_size`
       - Any generic hyperparameters you want as defaults.
     - Database, MongoDB, Kaggle, MLflow, and time-series hints (`TimeSeriesConfig`).
   - Respects environment variables:
     - `ML_TABULAR_ENV` to choose profile.
     - `ML_TABULAR_CONFIG_PATH` to override config file location.

2) **Run-specific training configs** (YAML under `configs/`)

   Examples:
   - `configs/train_tabular_baseline.yaml`
   - `configs/train_ts_baseline.yaml`

   These typically define:

   - Data paths / filenames:
     - `train_filename`, `valid_filename`, `test_filename`
   - Column semantics:
     - `target_column`
     - `feature_columns` (optional, or derived by script)
     - `categorical_columns`, `numeric_columns`
   - Model hyperparameters:
     - For tabular MLP:
       - Layer sizes, dropout rates, activation, etc.
     - For time series:
       - Input window length, prediction horizon.
   - Training hyperparameters:
     - Batch size, number of epochs, learning rate, optimizer choices.
   - Evaluation settings:
     - Metrics to compute (e.g. RMSE, MAE, accuracy).
   - MLflow:
     - Optional overrides like `experiment_name`, `run_name`.

The root training scripts and/or CLI commands read these YAML files, merge with `AppConfig` defaults, and pass the resulting config objects into the training utilities.


## 3. PyTorch datasets and models

### 3.1 Tabular: datasets and model

Module: `ml_tabular.torch.datasets.tabular`

Typical responsibilities:

- Take a **feature DataFrame** and:
  - Select predictor columns (`X`) and target column(s) (`y`).
  - Optionally apply basic transformations (e.g. standardization) if not handled in features or sklearn-based preprocessing.
- Convert to PyTorch tensors:
  - `X`: float tensor of shape `[N, num_features]`.
  - `y`: 1D or 2D tensor depending on the problem.
- Implement `__getitem__`/`__len__` so DataLoader can batch examples.

Module: `ml_tabular.torch.models.tabular_mlp`

- Defines a **multi-layer perceptron** suitable for tabular inputs:
  - Configurable input dimension.
  - Hidden layers (e.g. `[128, 64, 32]`).
  - Activation functions (e.g. ReLU).
  - Dropout (optional).
  - Final layer dimension:
    - 1 (regression) or number of classes (classification).

This model is intentionally generic, making it easy to:

- Swap in different architectures by subclassing or adding new model modules.
- Use the same training loop with different model instances.


### 3.2 Time series: dataset

Module: `ml_tabular.torch.datasets.time_series`

Typical responsibilities:

- Start from a **time-series DataFrame**, already:
  - Sorted by (series_id, timestamp).
  - Containing target and optional covariates.
- Build sliding windows:

  - For each series:
    - Use `input_window` (e.g. 30) and `prediction_horizon` (e.g. 7).
    - Construct input sequences:
      - `X[i] = [x(t - input_window + 1) ... x(t)]`
    - Construct targets:
      - `y[i]` might be:
        - next single step.
        - next horizon of steps (vector).

- Convert windows to PyTorch tensors.
- Keep track of series/indices if needed for evaluation.

The actual architecture for time series (MLP/BiLSTM/Transformer/etc.) can be varied while reusing the same dataset logic.


## 4. Training loops (torch/training/loops.py)

Module: `src/ml_tabular/torch/training/loops.py`

The idea is to centralize common deep-learning training boilerplate:

- Training over epochs:
  - Loop over DataLoader batches.
  - Forward pass → compute loss.
  - Backward pass → optimizer step.
- Validation:
  - Evaluate on validation DataLoader each epoch.
  - Compute metrics (RMSE, MAE, accuracy, etc.), depending on task.
- Logging:
  - Use `logging` (via `logging_config`) to report:
    - Epoch numbers.
    - Training and validation loss.
    - Metrics.
- Optional features:
  - Gradient clipping.
  - Early stopping:
    - Monitor a key metric or validation loss.
    - Stop after `patience` epochs without improvement.
  - Checkpointing:
    - Save best model state to `paths.models_dir` along with config snapshot.

The loops are written in a way that:

- Can be reused across multiple models (tabular vs time series).
- Support CPU/GPU via device parameter (and `model.to(device)`).
- Allow MLflow integration (if enabled).


## 5. Scikit-learn pipelines (sklearn_utils/tabular_pipeline.py)

Module: `src/ml_tabular/sklearn_utils/tabular_pipeline.py`

This module connects:

- **Stateless feature engineering** (from `ml_tabular.features.tabular`).
- **Scikit-learn preprocessing** and estimation.

Typical design:

- A `TabularPipelineBuilder` class with methods to build, configure, and retrieve a scikit-learn pipeline.
- Components:
  - Column selectors / transformers:
    - For numeric columns: standard scaling or other numeric transformers.
    - For categorical columns: OneHotEncoder/OrdinalEncoder, etc.
  - Optional custom transformers:
    - Wrapper around `build_features` if you want to incorporate feature engineering inside the pipeline.
  - Estimator:
    - Could be a RandomForest, GradientBoosting, XGBoost, etc., depending on config.

Benefits:

- Keeps the same tabular template usable for:
  - Classic ML baselines (quick to train).
  - Deep-learning models.
- Improves reproducibility:
  - Pipeline can be persisted as a single object (e.g. using joblib).
- Great for baseline creation and comparison:
  - You can have a baseline pipeline run defined in `train_tabular_baseline.yaml` alongside your deep learning runs.


## 6. Root training scripts

Two main scripts live at the project root:

1) `train_tabular_mlp.py`
2) `train_ts_mlp.py`

They are designed to:

- Parse command-line arguments:
  - Typically:
    - `--config` (path to YAML).
    - `--env` (environment name, aligning with `ML_TABULAR_ENV`).
    - Possibly overrides (like `--epochs`, `--batch-size`).

- Load:
  - Global `AppConfig` from `ml_tabular.config` (for paths, MLflow defaults).
  - Training run config from the provided YAML.

- Set up:
  - Logging (via `logging_config.configure_logging`).
  - Random seeds for reproducibility.
  - Device (CPU vs GPU).

- Run training:
  - For tabular:
    - Load data via `ml_tabular.data.loading`/`sql`/`mongodb`/`kaggle`.
    - Apply tabular features.
    - Create `TabularDataset` and DataLoader(s).
    - Build `TabularMLP`.
    - Call training loops with the dataset and model.
  - For time series:
    - Load time series data.
    - Apply time-series specific features.
    - Create `TimeSeriesDataset` and DataLoader(s).
    - Use `TabularMLP` or other model to train on windowed sequences.

- Handle MLflow (if enabled):
  - Start a run and log:
    - Run config (config YAML, AppConfig snapshot).
    - Training/validation metrics.
    - Artifacts (model state dict, plots, etc.).


## 7. CLI interface (ml_tabular.cli)

Module: `src/ml_tabular/cli.py`

The CLI provides a more user-friendly way to invoke training and utilities, using [Typer]-style commands (conceptually; actual dependency is defined in `pyproject.toml`).

Typical commands:

- `ml-tabular train-tabular-mlp --config configs/train_tabular_baseline.yaml`
- `ml-tabular train-ts-mlp --config configs/train_ts_baseline.yaml`
- Possibly:
  - `ml-tabular download-kaggle-dataset --dataset <id>`
  - `ml-tabular show-config --env dev`

The CLI:

- Wraps the root training script logic into functions.
- Calls `configure_logging` early.
- Finds and loads configuration files.
- Ensures that any exceptions are:
  - Logged using the standardized logging configuration.
  - Re-raised or reported with useful messages.

This lets you run everything with a single command and integrate with tools like `make`, CI, or external orchestration.


## 8. MLflow utilities (ml_tabular.mlops.mlflow_utils)

Module: `src/ml_tabular/mlops/mlflow_utils.py`

Purpose:

- Provide a thin abstraction over MLflow so your training scripts do not contain repetitive MLflow boilerplate.

Typical utilities:

- `start_run_with_config(cfg, run_config)`
  - Reads `cfg.mlflow` (from `AppConfig`):
    - `enabled`, `tracking_uri`, `experiment_name`, `run_name`, `log_artifacts`.
  - If `enabled`:
    - Sets tracking URI.
    - Ensures experiment exists or creates it.
    - Starts an MLflow run with the right name and tags.
  - Optionally logs:
    - Configuration YAML(s).
    - `AppConfig` as JSON.
    - Model hyperparameters.

- Helper functions to log:
  - Metrics (e.g. `log_metrics`).
  - Parameters (`log_params`).
  - Artifacts (e.g. saved model files, plots, data profiles).
  - Best epoch summary.

The training loops can be instrumented to:

- Call MLflow logging functions on each epoch.
- Log final model artifacts at the end of training.


## 9. Putting it all together

Example end-to-end tabular deep-learning run:

1) Prepare configuration:
   - Edit `configs/train_tabular_baseline.yaml` to specify:
     - Train/valid/test files or SQL sources.
     - Target and feature columns.
     - MLP architecture and hyperparameters.
     - Training hyperparameters (epochs, batch size, etc.).
     - MLflow settings if needed.

2) Set environment variables as needed:
   - `ML_TABULAR_ENV=dev`
   - `ML_TABULAR_CONFIG_PATH=config.yaml` (if you have a global config file).
   - Database / Mongo / Kaggle credentials where applicable.

3) Run:
   - `ml-tabular train-tabular-mlp --config configs/train_tabular_baseline.yaml`

4) Behind the scenes:
   - Logging is configured.
   - Config is loaded and validated.
   - Data is fetched and validated.
   - Features are built.
   - Dataset and DataLoader are created.
   - Model is constructed and trained via the central training loop.
   - Metrics and artifacts are logged (optionally to MLflow).

Example end-to-end time-series deep-learning run:

1) Configure `configs/train_ts_baseline.yaml`:
   - Include:
     - `datetime_column`, `target_column`, `group_column` (optional).
     - `input_window`, `prediction_horizon`.
     - Model hyperparameters.
     - MLflow info.

2) Run:
   - `ml-tabular train-ts-mlp --config configs/train_ts_baseline.yaml`

3) The script:
   - Loads time-series data.
   - Ensures correct sorting and time-series features.
   - Builds `TimeSeriesDataset` windows.
   - Trains the model using the training loop.
   - Logs run details and metrics.


## 10. Extending training and pipelines

Because the training and pipeline logic is modular:

- To add a new model:
  - Create a module under `ml_tabular.torch.models`.
  - Ensure it follows the same generic interface (e.g. `forward(x)` for tabular inputs).
  - Wire it into the training script or CLI via a config field (e.g. `model.type: "wide_deep"`).

- To add custom training behaviors:
  - Extend functions in `torch/training/loops.py`:
    - Custom callbacks/hooks.
    - Additional metrics.
    - Different schedulers or mixed-precision training.

- To add new pipelines:
  - Implement new builders or helpers in `sklearn_utils` or a new `pipelines` module.
  - Configure them through new YAML files.

The key philosophy:

- **Keep scripts thin**.
- **Push logic into reusable modules**.
- **Drive behavior via configuration**, so you can run multiple experiments and models without rewriting code each time.
