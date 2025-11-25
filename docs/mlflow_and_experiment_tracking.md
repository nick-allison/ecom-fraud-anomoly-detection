# MLflow and Experiment Tracking

This document explains how the `ml_tabular` template integrates with **MLflow** for experiment tracking, and how you can use it to keep your deep-learning and tabular/time-series experiments organized and reproducible.

The goals are:

- Make it **easy** to turn MLflow on/off per project or environment.
- Keep MLflow usage **centralized** in `ml_tabular.mlops.mlflow_utils`.
- Ensure experiments are **reproducible**:
  - Configs are logged.
  - Code version is captured (if available).
  - Metrics, parameters, and artifacts are stored consistently.


## 1. Where MLflow fits in the template

MLflow appears in three main places:

1) Configuration
   - `src/ml_tabular/config.py`
   - `MlflowConfig` section under `AppConfig`:
     - `enabled`
     - `tracking_uri`
     - `experiment_name`
     - `run_name`
     - `log_artifacts`

2) MLOps utilities
   - `src/ml_tabular/mlops/mlflow_utils.py`
   - Contains helper functions like:
     - `is_mlflow_available()`
     - `get_mlflow_client(cfg)`
     - `start_run_with_config(cfg, run_cfg, run_name)`
     - `log_config(cfg, run_cfg)`
     - `log_metrics(metrics, step)`
     - `log_artifacts_from_paths(paths, artifact_subdir)`

3) Training scripts / CLI commands
   - `train_tabular_mlp.py`
   - `train_ts_mlp.py`
   - `ml_tabular.cli` commands
   - These use mlflow_utils to:
     - Start runs.
     - Log metrics during training.
     - Save models and configs as artifacts.


## 2. Configuration: `MlflowConfig` in `AppConfig`

In `src/ml_tabular/config.py`, the MLflow-related configuration is defined as:

- `enabled: bool`
  - Global on/off switch for MLflow tracking.
  - Default: `False`.
  - Controlled via:
    - YAML config (e.g., `config.yaml`).
    - Environment variables:
      - `ML_TABULAR_MLFLOW__ENABLED=true`.

- `tracking_uri: str | None`
  - Where MLflow logs runs:
    - Local directory (default: `./mlruns`).
    - Remote tracking server (e.g. HTTP URL).
    - Databricks or other supported backends.
  - When `None`, MLflow uses its own default (local `mlruns`).

- `experiment_name: str | None`
  - Logical grouping of runs.
  - If set:
    - `mlflow.set_experiment(experiment_name)` is called.
  - If `None`:
    - The training script may choose a default (e.g., `"tabular_baselines"`).

- `run_name: str | None`
  - Optional default run name template.
  - Training scripts can:
    - Use this as a base.
    - Append dataset/model details:
      - e.g. `"tabular_mlp__titanic__2025-01-01"`.

- `log_artifacts: bool`
  - Indicates whether training scripts **should** log artifacts like:
    - Config YAML.
    - Model files.
    - Plots.
    - Metrics dumps.
  - This is a hint; code can still decide what to log.


### 2.1 Example MLflow section in `config.yaml`

The main config file might contain:

- Flat config:
  - `mlflow:`
    - `enabled: true`
    - `tracking_uri: "file:./mlruns"`
    - `experiment_name: "ml_tabular_experiments"`
    - `log_artifacts: true`

- Or environment-specific (dev/prod) profiles:
  - `dev:`
    - `mlflow:`
      - `enabled: true`
      - `tracking_uri: "file:./mlruns"`
  - `prod:`
    - `mlflow:`
      - `enabled: true`
      - `tracking_uri: "http://your-mlflow-server:5000"`
      - `experiment_name: "ml_tabular_prod"`


## 3. Optional dependency pattern

MLflow is declared under an optional extra (e.g. `[project.optional-dependencies].mlops`):

- This means:
  - `pip install -e .[mlops]` is needed to use MLflow features.
- In code:
  - Import is usually done in a guarded way within `mlflow_utils`:
    - If MLflow is not installed, helper functions can:
      - Raise a meaningful `AppError` / `ModelError`, or
      - Log a warning and no-op, depending on how you want the behavior.


## 4. `ml_tabular.mlops.mlflow_utils` responsibilities

This module centralizes MLflow usage so that:

- Training code is clean and free of direct MLflow wiring.
- You can change MLflow behavior in one place (logging, tags, artifacts).

Common responsibilities:

1) Availability checks
   - `is_mlflow_available()`
     - Returns `True` if `mlflow` can be imported.
     - Used for defensive checks.

2) Client setup
   - `get_mlflow_client(cfg)`
     - Applies `cfg.mlflow.tracking_uri` if provided:
       - e.g. `mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)`.

3) Experiment selection
   - `set_experiment_from_config(cfg)`
     - Calls `mlflow.set_experiment(cfg.mlflow.experiment_name)` if set.
     - Otherwise uses a default experiment name, or MLflow’s default.

4) Run lifecycle helpers

   - Example pattern:

     - `start_run_with_config(cfg, run_cfg, run_name=None)`
       - Uses:
         - `cfg.mlflow.enabled` to decide whether to actually start a run.
         - `cfg.mlflow.run_name` as a fallback for `run_name`.
         - `cfg.mlflow.experiment_name` for experiment selection.
       - Returns:
         - A context manager or run handle so training code can do:
           - `with start_run_with_config(...) as run: ...`.

   - `log_config(cfg, run_cfg)`
     - Logs:
       - Effective app config (`cfg.to_dict()`).
       - Training run config (`run_cfg` loaded from YAML).
       - Possibly as:
         - `mlflow.log_params` (flattened).
         - `mlflow.log_dict` or `mlflow.log_artifact` with YAML.

   - `log_metrics(metrics: Mapping[str, float], step: int | None = None)`
     - Wraps `mlflow.log_metrics` and handles:
       - The case when MLflow is disabled.
       - Optional `step` argument.

   - `log_artifacts_from_paths(paths: Sequence[Path], artifact_subdir: str | None = None)`
     - Logs files/folders as artifacts.
     - Supports:
       - Config copies (e.g. `configs/train_tabular_baseline.yaml`).
       - Model checkpoints (e.g. `models/tabular_mlp.pth`).
       - Plots or reports created by training code.


## 5. How training scripts use MLflow

### 5.1 `train_tabular_mlp.py`

Conceptually, the script:

1) Loads config:
   - `cfg = get_config(env=env)`
   - `run_cfg = load_run_config(config_path)`

2) Configures logging.

3) If MLflow is enabled (`cfg.mlflow.enabled`):

   - Calls something like:
     - `with start_run_with_config(cfg, run_cfg, run_name=...) as run:`
       - Logs:
         - Experiment name.
         - Run name.
         - `cfg` and `run_cfg` (parameters).
       - Inside the context:
         - Training happens.
         - Metrics are logged at:
           - Epoch level.
           - Validation step level.

   - Example metric logging inside the training loop:

     - For each epoch:
       - `mlflow_utils.log_metrics({"train_loss": loss_value, "val_loss": val_loss}, step=epoch)`
       - `mlflow_utils.log_metrics({"val_accuracy": accuracy}, step=epoch)`

   - After training:
     - If `cfg.mlflow.log_artifacts`:
       - Save model checkpoint(s) locally.
       - Log them as MLflow artifacts:
         - `mlflow_utils.log_artifacts_from_paths([checkpoint_path], artifact_subdir="models")`
       - Log the config/run config YAML files as artifacts.

4) If MLflow is disabled:
   - Training still runs.
   - Logging utilities either:
     - No-op gracefully, or
     - Log a message saying MLflow is disabled.


### 5.2 `train_ts_mlp.py`

Follows the same patterns as tabular, but uses:

- Time-series-specific:
  - Dataset: `TimeSeriesDataset` (or equivalent).
  - Config: `train_ts_baseline.yaml` fields.
- MLflow helpers in exactly the same way.
  - Metrics:
    - `train_loss`, `val_loss`
    - Forecasting metrics (e.g. `val_mse`, `val_mae`, `val_mape`, etc.)
  - Artifacts:
    - Time series plots.
    - Forecast vs actual overlays.
    - Model files.


## 6. What gets tracked in MLflow

You can tailor what you log, but a solid baseline includes:

1) Parameters (via `log_params` or flattened dicts)
   - Model hyperparameters:
     - `hidden_dim`, `n_layers`, `dropout`, etc.
   - Training hyperparameters:
     - `lr`, `batch_size`, `epochs`.
   - Data choices:
     - `dataset_name`, `input_features`, `target_column`.

2) Metrics (via `log_metric` / `log_metrics`)
   - Training metrics:
     - `train_loss`, `train_accuracy`, etc.
   - Validation metrics:
     - `val_loss`, `val_rmse`, `val_r2`, etc.
   - Time-series metrics:
     - `val_mse`, `val_mae`, `val_mape`, etc.
   - Logged with a `step` argument (usually the epoch index).

3) Artifacts
   - Model checkpoints:
     - `models/tabular_mlp_epoch_{epoch}.pth`
     - Final model: `models/tabular_mlp_final.pth`
   - Config snapshots:
     - `configs/effective_app_config.yaml`
     - `configs/train_tabular_baseline.yaml`
   - Logs:
     - Training log file(s) from `logs/`.
   - EDA artifacts:
     - Plots saved during quickstart notebooks (if logged).

4) Tags (optional but recommended)
   - Static metadata:
     - `mlflow.set_tag("model_type", "tabular_mlp")`
     - `mlflow.set_tag("framework", "pytorch")`
     - `mlflow.set_tag("env", cfg.env)`
     - `mlflow.set_tag("dataset_name", run_cfg["data"]["dataset_name"])`
   - These can be set via a helper like:
     - `log_common_tags(cfg, run_cfg)` in `mlflow_utils`.


## 7. Local vs remote tracking

Because MLflow config is centralized, you can easily switch between:

1) Local tracking (default)
   - No `tracking_uri` set, or something like:
     - `file:./mlruns`
   - Creates a local `mlruns` directory.
   - Inspect runs via:
     - `mlflow ui --backend-store-uri ./mlruns`

2) Remote tracking server
   - `cfg.mlflow.tracking_uri` set to:
     - `http://mlflow.yourcompany.com:5000`
   - Data is stored in a shared backend store and artifact store.
   - Useful for:
     - Team collaboration.
     - Persisting production experiment history.

The training code does not need to change; it only obeys `AppConfig.mlflow`.


## 8. Reproducibility and experiment hygiene

The template aims to encourage good experiment hygiene:

1) Config-driven runs
   - Every run should be associated with:
     - A global config profile (dev/prod).
     - A run-specific YAML (e.g. `train_tabular_baseline.yaml`).
   - These configs should be:
     - Version-controlled.
     - Logged as artifacts in MLflow.

2) Code versioning
   - When using git, MLflow can optionally:
     - Capture the git commit, branch, and dirty state.
   - You can integrate this into `mlflow_utils.start_run_with_config`:
     - Use `mlflow.set_tag("git_commit", <commit_hash>)` if available.

3) Names and conventions
   - Experiments:
     - `ml_tabular_tabular_baselines`
     - `ml_tabular_time_series_baselines`
   - Run names:
     - `tabular_mlp__dataset=titanic__seed=42`
     - `ts_mlp__dataset=traffic_volume__input=48__horizon=12`

4) Avoid ad-hoc runs
   - Even quick tests should ideally use:
     - A small run config file, or
     - A dedicated “scratch” experiment (e.g. `"dev_scratch"`).


## 9. Using MLflow UI to explore runs

Once you have runs:

1) Start the UI:

   - If local:
     - `mlflow ui --backend-store-uri ./mlruns`
   - If remote:
     - UI is typically running at your tracking server URL.

2) Select the experiment for:
   - Tabular models.
   - Time-series models.

3) Compare runs:
   - Filter by tags (model type, dataset).
   - Compare metrics:
     - `val_loss`, `val_rmse`, etc.
   - Sort by performance.

4) Drill into a run:
   - Inspect:
     - Parameters.
     - Metrics over time.
     - Artifacts:
       - Model checkpoint.
       - Plots.
       - Config files.


## 10. Extending MLflow integration

You can extend MLflow integration as your needs grow:

1) Model registry
   - Register best models to the MLflow Model Registry.
   - Introduce helpers in `mlflow_utils` like:
     - `register_best_model(run_id, model_uri, registered_model_name)`.

2) Deployment integration
   - Use logged models for:
     - Batch scoring jobs.
     - Online serving (e.g. MLflow serving, custom FastAPI, etc.).

3) Advanced tracking
   - Log:
     - Confusion matrices.
     - ROC curves.
     - Feature importance plots.
   - Use `mlflow.log_figure` or log images as artifacts.

4) Automated hyperparameter search
   - Integrate with frameworks like:
     - Optuna.
     - Ray Tune.
   - Use MLflow to track:
     - Each trial as a separate run.
     - Best hyperparameter configurations.


## 11. Summary

The MLflow integration in `ml_tabular` is designed so that:

- You can **turn it on/off** via configuration (`cfg.mlflow.enabled`).
- All MLflow-related logic is **centralized** in `ml_tabular.mlops.mlflow_utils`.
- Training scripts for:
  - Tabular models (`train_tabular_mlp.py`)
  - Time-series models (`train_ts_mlp.py`)
  - …can all share consistent tracking patterns.

This gives you:

- A professional, production-ready approach to:
  - Experiment tracking.
  - Result comparison.
  - Reproducibility.
- A strong story to tell in interviews and portfolio:
  - Your template isn’t just about model code—it treats experiments as first-class citizens.
