# CLI and Scripts

This document explains how the **command-line interface (CLI)** and top-level **scripts** are structured in the `ml_tabular` template.

The goals of the CLI and scripts are:

- Provide a single, predictable entry point (`ml-tabular`) for most operations.
- Make it easy to run:
  - Training for tabular models.
  - Training for time-series models.
  - Utility commands (Kaggle, config inspection, etc.).
- Keep logic **in library code**, not sprinkled across ad-hoc scripts.


## 1. Overview

There are two main layers:

1) Python package CLI (preferred)
   - Module: `src/ml_tabular/cli.py`
   - Exposed as a console script:
     - Command: `ml-tabular`
   - Uses a Typer-style interface (friendly help, subcommands, type-checked options).

2) Root-level scripts (thin wrappers)
   - Files at project root:
     - `train_tabular_mlp.py`
     - `train_ts_mlp.py`
   - These are intentionally very small:
     - They parse arguments.
     - They call into library functions / CLI logic.


## 2. The `ml-tabular` console command

Defined in `pyproject.toml` (project.scripts section):

- Script name:
  - `ml-tabular`
- Python entry-point:
  - `ml_tabular.cli:app` (Typer application instance).

Once the project is installed (e.g. `pip install -e .` in a virtual environment), you can use:

- `ml-tabular --help`
- `ml-tabular train-tabular-mlp --help`
- `ml-tabular train-ts-mlp --help`
- Etc.

The exact set of subcommands depends on how `cli.py` is implemented, but the template is designed to support at least:

- `train-tabular-mlp`
- `train-ts-mlp`
- Utility commands (Kaggle, config inspection) when added.


## 3. CLI module: `ml_tabular.cli`

Conceptual responsibilities:

1) Central entry-point for the package
   - Defines a Typer app (e.g., `app = typer.Typer()`).
   - Decorates functions as subcommands.

2) Shared concerns for all commands
   - Logging:
     - Calls `logging_config.configure_logging` early.
   - Configuration:
     - Provides a consistent way to load `AppConfig` and YAML configs.
   - Error handling:
     - Catches known exceptions (e.g. `AppError`, `ConfigError`, `DataError`).
     - Logs them with context.
     - Exits with appropriate status codes.

3) Command categories (typical)

   - Training
     - `train-tabular-mlp`
     - `train-ts-mlp`
   - Data
     - `download-kaggle-dataset` (optional).
   - Diagnostics / utility
     - `show-config`
     - `check-env`
     - `list-config-profiles`

Internally, each command usually follows the pattern:

- Accept simple command-line options:
  - `--config` (path to run YAML config).
  - `--env` (environment profile name).
  - Override flags like `--epochs`, `--batch-size` if needed.

- Use shared helpers:
  - Load YAML into a dict.
  - Load `AppConfig` via `ml_tabular.config.get_config`.
  - Configure logging.
  - Call into a “runner” function in another module (e.g., training runner).


## 4. Training commands via CLI

### 4.1 Tabular MLP training

Subcommand: conceptually `ml-tabular train-tabular-mlp`

Typical options:

- `--config PATH`
  - Path to `configs/train_tabular_baseline.yaml` (or another run config).
- `--env TEXT`
  - Overrides `ML_TABULAR_ENV` to select a profile in the global config.
- Optional overrides:
  - `--epochs INT`
  - `--batch-size INT`
  - `--learning-rate FLOAT`
  - etc.

High-level flow in the command function:

1) Configure logging.
2) Load `AppConfig`:
   - `cfg = get_config(env=env)`.
3) Load run config YAML:
   - e.g., `run_cfg = load_yaml(config_path)`.
4) Infer paths and filenames from:
   - `cfg.paths`
   - run config fields.
5) Load data:
   - `ml_tabular.data.loading.load_raw_dataset` or SQL / Mongo / Kaggle helpers.
6) Apply tabular features:
   - `ml_tabular.features.tabular.build_features`.
7) Build PyTorch dataset:
   - `TabularDataset`.
8) Build model:
   - `TabularMLP`.
9) Run training loop:
   - `train_model` from `torch/training/loops.py`.
10) Handle MLflow logging:
    - If `cfg.mlflow.enabled` is true, start run, log metrics and artifacts.

In practice, most of the above is delegated to helper functions so the CLI command function remains small and readable.


### 4.2 Time-series MLP training

Subcommand: conceptually `ml-tabular train-ts-mlp`

Typical options:

- `--config PATH`
  - Path to `configs/train_ts_baseline.yaml`.
- `--env TEXT`
  - Environment profile name.
- Optional overrides:
  - `--epochs`, `--batch-size`, etc.

Flow:

1) Configure logging and load configs (same as tabular).
2) Load time-series data from file, SQL, or other source.
3) Apply time-series feature engineering:
   - `ml_tabular.features.time_series.build_time_series_features` (or equivalent).
4) Build `TimeSeriesDataset`:
   - Uses window length (`input_window`) and `prediction_horizon` from config.
5) Build model:
   - For example, MLP on flattened windows, or a more specialized model.
6) Use training loops for training and evaluation.
7) Optionally log everything to MLflow.


## 5. Root scripts: `train_tabular_mlp.py` and `train_ts_mlp.py`

Even though the CLI is the preferred interface, the template also includes:

- `train_tabular_mlp.py`
- `train_ts_mlp.py`

Reasons to keep these:

1) Explicit discoverability
   - Many ML developers are used to seeing `train_*.py` at the project root.
   - Recruiters / reviewers can quickly spot that there are dedicated train scripts.

2) Integration with notebooks and quick experiments
   - From a notebook or shell:
     - `!python train_tabular_mlp.py --config configs/train_tabular_baseline.yaml`
     - etc.

3) Thin wrappers, not logic containers
   - They should:
     - Parse `sys.argv` (often via `argparse` or Typer itself).
     - Call the same underlying functions or CLI entry that `ml-tabular` uses.

A typical pattern:

- The script imports `main_tabular_mlp` (or similar) from `ml_tabular.cli` or a dedicated runner module.
- The `if __name__ == "__main__":` block simply calls that function with parsed arguments.


## 6. Utility commands (Kaggle, configuration, etc.)

The template is designed so it is easy to add new CLI commands for:

1) Kaggle integration (optional)
   - `download-kaggle-dataset`
     - Uses the `kaggle` Python API.
     - Reads defaults from `cfg.kaggle` (e.g., dataset id, competition id).
     - Downloads data into `paths.raw_dir / cfg.kaggle.download_subdir`.

   Typical options:
   - `--dataset TEXT`
   - `--competition TEXT`
   - `--output-dir PATH`
   - `--force` to overwrite.

2) Config inspection
   - `show-config`
     - Prints the effective `AppConfig` (merged from YAML + env).
   - `show-paths`
     - Prints resolved paths from `cfg.resolved_paths()`.

3) Environment checks
   - `check-env`
     - Validates presence of environment variables for:
       - Database connection.
       - Kaggle API token.
       - MLflow tracking URI.

These commands live in `ml_tabular.cli` and simply call appropriate helpers from modules like `ml_tabular.config`, `ml_tabular.data`, and `ml_tabular.mlops`.


## 7. Error handling and exit codes

The CLI and scripts are designed to handle errors in a controlled way:

- Exceptions (e.g. `ConfigError`, `DataError`, `ModelError`, `PipelineError`) are:
  - Logged using a structured message:
    - Include error `code`, `location`, and `context`.
  - Converted into user-friendly terminal output.

- Exit codes:
  - Successful run: exit code 0.
  - Known application errors:
    - Use non-zero codes (e.g. `1`, `2`) depending on how you choose to map error types.
  - Unexpected errors:
    - Logged with stack traces if configured for debug environments.

This aligns with good practice for:
- CI pipelines (where exit codes matter).
- Integration with orchestration tools (Airflow, Prefect, etc.).


## 8. Development workflow with CLI and scripts

A typical development loop:

1) Install the project in editable mode:
   - `pip install -e .[dev,mlops]`

2) Run basic checks:
   - `pre-commit run --all-files`
   - `pytest`

3) Use CLI for training:
   - `ml-tabular train-tabular-mlp --config configs/train_tabular_baseline.yaml`
   - `ml-tabular train-ts-mlp --config configs/train_ts_baseline.yaml`

4) Use utility commands as needed:
   - `ml-tabular show-config --env dev`
   - `ml-tabular download-kaggle-dataset --dataset <id>`

The root scripts remain available for:
- Quick, explicit command-line and notebook invocations.
- Fallback if console-script wiring changes (you can always run `python train_tabular_mlp.py ...`).


## 9. Extending the CLI and scripts

Because most logic is in modular functions, extending the CLI is straightforward:

1) Add a new function in `ml_tabular.cli`:
   - Decorate with `@app.command("new-command-name")`.
   - Accept typed parameters (e.g. `config: Path`, `env: str`).

2) Implement the body by calling into:
   - Existing utility modules (data, config, features, training).
   - Or new modules you add (e.g. `ml_tabular.torch.models.custom_model`).

3) Optionally:
   - Add documentation in `docs/cli_and_scripts.md` (this file).
   - Add tests in `tests/cli/test_cli.py`.

Best practices:

- Keep CLI functions thin:
  - No heavy logic in `cli.py` itself.
  - Route to helper “runner” modules/functions.
- Maintain a consistent naming scheme for commands:
  - `train-*`, `download-*`, `show-*`, `check-*` etc.
- Ensure error handling is consistent:
  - Catch `AppError` subclasses.
  - Use `logger.exception` for unexpected exceptions when `env` is “dev”.


## 10. Summary

The CLI and scripts in `ml_tabular` are designed to:

- Provide a clean, single command (`ml-tabular`) to run:
  - Tabular training.
  - Time-series training.
  - Data and MLOps utilities.
- Keep training logic reusable and testable by locating it in:
  - `ml_tabular.torch.datasets.*`
  - `ml_tabular.torch.models.*`
  - `ml_tabular.torch.training.loops`
  - `ml_tabular.sklearn_utils.tabular_pipeline`
  - `ml_tabular.mlops.mlflow_utils`
- Support both:
  - Advanced deep-learning workflows.
  - Classic ML baselines and quick experiments.

By following this structure, you get a project where:

- Onboarding is simple (`ml-tabular --help` shows everything).
- Experiments are consistently run via explicit configs and commands.
- Logic is properly modularized and testable.
