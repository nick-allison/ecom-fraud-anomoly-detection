# Time-Series Quickstart

This guide explains how to run end-to-end time-series experiments with the ml_tabular structured template:

- Load time-series data from files, SQL, Mongo, or Kaggle
- Build consistent time-based features and sliding windows
- Train deep-learning models (e.g. MLP over windows) with PyTorch
- (Optionally) track experiments with MLflow

Use this as your reference when starting any time-series project from this template.


## 1. Prerequisites

1) Install the project in editable mode:

   - Create and activate a virtual environment
   - From the repo root:

     pip install -e ".[dev,mlops,validation]"

   This gives you:
   - Core ML stack (numpy, pandas, scikit-learn, torch, pydantic, etc.)
   - Developer tooling (ruff, mypy, pytest, pre-commit)
   - Optional MLflow + Kaggle integration
   - Optional pandera for extra validation

2) (Optional but recommended) Install pre-commit hooks:

   pre-commit install

   This keeps your code quality high (linting, formatting, basic checks) on every commit.


## 2. Configuration overview (time-series focus)

Time-series experiments are configured via:

- src/ml_tabular/config.py              (global configuration schema + loader)
- configs/train_ts_baseline.yaml        (experiment-specific time-series settings)

The AppConfig model includes a dedicated TimeSeriesConfig section, which exposes fields like:

- datetime_column      : the main timestamp column
- target_column        : the value you’re forecasting
- group_column         : optional ID for multiple related series
- freq                 : nominal frequency (e.g. "D", "H")
- input_window         : length of the input sequence per example
- prediction_horizon   : how many steps ahead to forecast
- max_series_length    : optional cap on series length
- normalize_per_series : whether to normalize each series individually

These global hints are complemented by the more experiment-specific settings inside train_ts_baseline.yaml (e.g. file names, feature columns, model hyperparameters).


## 3. The baseline time-series config

The file configs/train_ts_baseline.yaml encodes a “standard” time-series setup, conceptually similar to:

env: "dev"
experiment_name: "ts_baseline"

paths:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  models_dir: "models"

data:
  train_file: "ts_train.csv"
  valid_file: "ts_valid.csv"
  test_file: "ts_test.csv"
  datetime_column: "timestamp"
  target_column: "target"
  group_column: "series_id"
  freq: "D"

  extra_covariates:
    - "temperature"
    - "promo_flag"

  # Optional: which columns are allowed as numeric vs categorical covariates
  numeric_features:
    - "temperature"
  categorical_features:
    - "promo_flag"

windows:
  input_window: 30
  prediction_horizon: 7
  max_series_length: null
  normalize_per_series: true

features:
  datetime_features:
    - "dayofweek"
    - "month"
    - "is_weekend"

model:
  type: "mlp"
  hidden_dims: [128, 64]
  dropout: 0.1
  activation: "relu"
  task_type: "regression"

training:
  batch_size: 256
  max_epochs: 25
  learning_rate: 1e-3
  weight_decay: 1e-4
  num_workers: 4
  random_seed: 42

mlflow:
  enabled: true
  experiment_name: "time_series_baselines"

You will customize this structure for your own datasets (file names, columns, window sizes, etc.).


## 4. Time-series data layout and loading

### 4.1 Data shape and columns

The baseline template expects your time-series data in a “long” format, for example:

series_id,timestamp,target,temperature,promo_flag
A,2020-01-01,10.5,25.1,0
A,2020-01-02,11.2,24.9,1
...
B,2020-01-01,5.3,30.0,0
...

Key expectations:

- datetime_column (e.g. "timestamp"):
  - Can be parsed into a datetime
  - Defines the ordering of observations

- target_column (e.g. "target"):
  - The value to forecast

- group_column (e.g. "series_id"):
  - Optional; if present, each distinct value represents a separate series

Other columns:

- numeric covariates (e.g. "temperature")
- categorical covariates (e.g. "promo_flag")


### 4.2 Loading the raw data

As with tabular, you can start from:

- CSV/Parquet/Feather files in data/raw
- SQL tables (via ml_tabular.data.sql)
- Mongo collections (via ml_tabular.data.mongodb)
- Kaggle downloads (via ml_tabular.data.kaggle)

In the baseline script train_ts_mlp.py, the typical pattern is:

- Get paths from config: get_paths()
- Load train/valid/test DataFrames from raw_dir or processed_dir
- Ensure columns specified in the YAML exist (datetime, target, group, covariates)


## 5. Time-series feature engineering and windowing

Time-series has two big steps:

- Per-row feature engineering (e.g. dayofweek, month, is_weekend)
- Converting sequences of rows into sliding windows suitable for deep learning


### 5.1 Per-row time features

This is handled by:

- ml_tabular.features.time_series

You can configure which datetime-based features to create, for example:

- dayofweek
- month
- hour
- is_weekend
- [Your custom choices]

Internally, the code:

1) Parses the datetime column into a proper pd.DatetimeIndex
2) Adds new columns such as:
   - ts__dayofweek
   - ts__month
   - ts__is_weekend
   - etc.
3) Optionally normalizes or scales covariates

The goal is to keep this logic:

- Explicit
- Deterministic
- Version-controlled (through your Python and YAML files)


### 5.2 Sliding windows for deep learning

The “heart” of the time-series setup is converting ordered sequences into:

- Input windows of length input_window
- Targets covering prediction_horizon steps ahead

This is handled by:

- ml_tabular.torch.datasets.time_series.TimeSeriesDataset

Conceptually, TimeSeriesDataset:

1) Groups the DataFrame by group_column (or treats it as a single global series if group_column is None)
2) Sorts each series by datetime_column
3) Applies optional truncation with max_series_length
4) Normalizes each series individually if normalize_per_series is True
5) Slides a window of length input_window across each series, and for each window:
   - Extracts:
     - X: past target values (and/or covariates)
     - y: the next prediction_horizon values of the target
   - Stores these as PyTorch tensors

This yields a dataset where each index corresponds to:

- One (possibly multivariate) input window
- Matching target horizon

The DataLoader then batches these windows for training and validation.


## 6. Time-series model and training loop

### 6.1 Model

The baseline time-series model is typically implemented in:

- ml_tabular.torch.models.tabular_mlp (or a specialized ts_model, depending on your implementation)

Even though it’s called “tabular,” the underlying idea is simple:

- Flatten each window into a single feature vector
- Use a multilayer perceptron to map from that vector to the forecast horizon

For example:

- input_dim  = input_window * (num_targets + num_covariates)
- hidden_dims, dropout, activation as configured in the YAML
- output_dim = prediction_horizon (for a univariate target)
- task_type  = "regression" (most forecasting problems)

You can later swap this out for more advanced architectures, such as:

- 1D CNNs over time
- LSTMs / GRUs / Transformers
- Specialized forecasting networks

The baseline MLP is intentionally simple, so the template demonstrates structure without locking you into a single architecture.


### 6.2 Training loop

The training logic is shared with tabular and lives in:

- ml_tabular.torch.training.loops

Typical functions include:

- train_epoch(model, dataloader, optimizer, loss_fn, device, ...)
- evaluate(model, dataloader, loss_fn, device, metrics, ...)

train_ts_mlp.py orchestrates:

1) Configure logging
2) Load AppConfig and YAML config
3) Load and preprocess DataFrames (including time features)
4) Build TimeSeriesDataset and DataLoaders for train/valid
5) Instantiate the model and optimizer
6) Train for max_epochs:
   - Log metrics per epoch
   - Optionally track the best validation score and save best weights
7) Evaluate on test set if configured
8) Save the final or best model under models_dir


### 6.3 Metrics

Common regression metrics for time-series forecasting include:

- MSE / RMSE
- MAE
- MAPE (watch out for zeros)
- custom domain-specific metrics

The training script can compute and log these metrics at the end of:

- Each epoch (on validation data)
- Final evaluation (on test data)

If MLflow is enabled, metrics are also logged to your tracking server.


## 7. MLflow integration (optional but recommended)

If mlflow.enabled is true in config:

- ml_tabular.mlops.mlflow_utils takes care of:
  - Setting the tracking URI (local or remote)
  - Ensuring an experiment exists (experiment_name)
  - Starting a run
  - Logging:
    - Hyperparameters (window size, horizon, hidden_dims, etc.)
    - Loss and metrics per epoch
    - Artifacts (model weights, configs, plots)

This allows you to:

- Compare multiple window sizes or horizons
- Compare architectures (MLP vs RNN vs Transformer)
- Track performance across datasets and preprocessing variations


## 8. Running the baseline time-series experiment

### 8.1 Using the training script directly

From the repo root, run:

python train_ts_mlp.py --config configs/train_ts_baseline.yaml

Typical flow inside the script:

1) Parse arguments (e.g. config path, optional overrides)
2) Configure logging
3) Load configuration (AppConfig + experiment YAML)
4) Load DataFrames for train/valid/test
5) Build time-based features
6) Build windows via TimeSeriesDataset
7) Construct the model
8) Train and evaluate
9) Save model and optionally log MLflow run


### 8.2 Using the CLI

If you’ve wired up a CLI in:

- src/ml_tabular/cli/app.py

Then the time-series command might look like:

ml-tabular train-ts --config configs/train_ts_baseline.yaml

Advantages:

- A consistent command-line entry point for both tabular and time-series experiments
- Easy to integrate into CI/CD or scheduled jobs
- Simple place to add future commands like:
  - ml-tabular evaluate-ts
  - ml-tabular predict-ts
  - ml-tabular export-ts-model


## 9. Customizing time-series experiments

To adapt the baseline to your own data:

1) Copy configs/train_ts_baseline.yaml to something like:

   configs/train_ts_electric_load.yaml

2) Edit:

   - data.*:
     - train_file, valid_file, test_file
     - datetime_column, target_column, group_column
     - numeric_features, categorical_features (if any)
   - windows.*:
     - input_window
     - prediction_horizon
     - max_series_length
     - normalize_per_series
   - features.*:
     - Which datetime features to add
     - Any additional per-row transformations
   - model.*:
     - hidden_dims, dropout, activation, task_type
     - model type if you introduce RNN/Transformer variants
   - training.*:
     - batch_size, max_epochs, learning_rate, etc.
   - mlflow.*:
     - experiment_name, tracking_uri

3) Run:

   python train_ts_mlp.py --config configs/train_ts_electric_load.yaml

4) Iterate:
   - Try different window lengths and horizons
   - Change architectures
   - Add/modify covariates
   - Track everything via MLflow


## 10. Using time-series notebooks

The notebooks for time-series live under notebooks/ and include:

- 10_time_series_quickstart.ipynb
  - Minimal end-to-end example for one series
  - Shows how TimeSeriesDataset is constructed and how the baseline model behaves

- 11_time_series_eda_and_windows.ipynb
  - Exploratory analysis of the raw series
  - Visualizations of:
    - Seasonality, trends, outliers
    - Distribution of windowed inputs vs targets
  - Hands-on experimentation with:
    - Different input_window / prediction_horizon configurations
    - Normalization choices

You can use these notebooks to:

- Understand how the dataset and windows are formed
- Inspect whether features make sense (e.g. dayofweek, promos, holidays)
- Prototype new architectures before baking them into train_ts_mlp.py


## 11. Data sources: SQL, Mongo, Kaggle

Time-series often comes from operational systems — for example:

- SQL tables with event logs or sensor readings
- MongoDB collections with time-stamped documents
- Kaggle datasets for competitions or public benchmarks

The template supports this via:

- ml_tabular.data.sql:
  - load_sql_table(), load_sql_query(), execute_sql()
- ml_tabular.data.mongodb:
  - load_mongo_collection(), count_documents()
- ml_tabular.data.kaggle:
  - Download competition or dataset files into data/raw/kaggle/

Typical flow:

1) Use these helpers in a small ETL script or notebook to pull data into DataFrames.
2) Save cleaned outputs as CSV/Parquet in data/raw or data/processed.
3) Point train_ts_baseline.yaml at those files.

This keeps your time-series training code decoupled from the raw operational systems and credentials.


## 12. Summary

The time-series quickstart is designed to show that you can:

- Treat time-series as a first-class modality
- Manage configuration, paths, and logging cleanly
- Build reproducible windows and features
- Train deep-learning models in a structured way
- Track experiments using MLflow
- Integrate with realistic data sources (files, SQL, Mongo, Kaggle)

For new time-series projects:

1) Duplicate this template
2) Point configs/train_ts_baseline.yaml at your data
3) Customize TimeSeriesConfig and window parameters
4) Run train_ts_mlp.py
5) Iterate on features and models, keeping everything reproducible and traceable
