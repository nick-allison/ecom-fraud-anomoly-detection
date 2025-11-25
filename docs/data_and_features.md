# Data and Feature Engineering

This document explains how the **data layer** and **feature layer** work in the `ml_tabular` template.

The goals are to:

- Make it easy to pull data from multiple sources (files, SQL, MongoDB, Kaggle).
- Keep feature engineering **stateless** and **testable**.
- Share patterns between **tabular** and **time-series** projects.
- Give you a clean place to add project-specific logic without turning your notebook into a giant script.


## 1. Big picture: data → features → datasets → models

The standard flow in this template is:

1) **Data loading**  
   - Use `ml_tabular.data.*` modules to get raw data into **pandas DataFrames**.
   - Support:
     - Local files (CSV, Parquet, Feather)
     - Relational databases (via SQLAlchemy)
     - MongoDB (optional)
     - Kaggle datasets and competitions (optional)

2) **Feature engineering**  
   - Use `ml_tabular.features.*` modules to create **stateless, deterministic** features.
   - For tabular:
     - Datetime expansion, log transforms, ratios, powers, interactions.
   - For time series:
     - Datetime-derived features that help sequence models.
     - Optional helpers to ensure consistent ordering before windowing.

3) **Datasets and modeling** (handled elsewhere)  
   - Use PyTorch datasets to turn feature DataFrames into tensors.
   - Train models using the training loops and scripts.

This separation lets you:

- Swap data sources without touching feature code.
- Change feature specs without rewriting loaders.
- Reuse the same feature logic for multiple models (baseline, MLP, etc.).


## 2. Data layer: ml_tabular.data

The data layer’s job is: **“given a configuration and a source, return a clean DataFrame or iterator of DataFrames.”**

It does **not**:

- Know about model architectures.
- Apply complex ML-specific transforms.
- Perform heavy feature engineering.

Instead, it focuses on:

- **I/O** (files, databases, APIs)
- **Basic validation** (e.g. required columns)
- **Safe logging** (never log secrets)


### 2.1 File-based loading: loading.py

Module: `src/ml_tabular/data/loading.py`

Core functions:

- `load_dataframe(path, format=None, required_columns=None, read_kwargs=None)`
- `load_raw_dataset(filename, paths=None, required_columns=None, read_kwargs=None)`
- `load_processed_dataset(filename, paths=None, required_columns=None, read_kwargs=None)`

Key behavior:

- Infers format from file suffix if not provided:
  - `.csv` → CSV
  - `.parquet` → Parquet
  - `.feather` → Feather
- Checks that the file exists; raises `DataError` if not.
- Reads with pandas using appropriate reader:
  - `pd.read_csv`, `pd.read_parquet`, `pd.read_feather`
- Optionally validates `required_columns`:
  - If any required columns are missing, raises `DataError` with:
    - list of missing columns
    - list of available columns
- Logs:
  - Path, format, row/column counts
  - Any issues (e.g. missing columns) via structured `DataError`.

`load_raw_dataset` and `load_processed_dataset`:

- Use `get_paths()` from `ml_tabular.config` to resolve:
  - `paths.raw_dir / filename`
  - `paths.processed_dir / filename`
- Let you keep your training scripts clean:

  - Instead of:
    - `pd.read_csv("data/raw/train.csv")`
  - You do:
    - `df_train = load_raw_dataset("train.csv", required_columns=[...])`

This centralizes file layout in config rather than scattering paths in code.


### 2.2 SQL loading: sql.py

Module: `src/ml_tabular/data/sql.py`

Core APIs:

- `get_engine(echo=None)`  
  - Builds a SQLAlchemy `Engine` using `AppConfig.database`.
  - Uses:
    - `database.url` (e.g. `"postgresql+psycopg://user:pass@host:5432/dbname"`)
    - `database.echo` (if `echo` not passed explicitly).
  - Logs only a **safe driver name** (like `"postgresql+psycopg"`) via `_safe_db_driver_name`, never the full URL.

- `load_sql_query(query, params=None, chunksize=None)`  
  - Executes an arbitrary SQL query:
    - If `chunksize is None`, returns a single DataFrame.
    - If `chunksize` is set, returns an iterator of DataFrames (streaming).
  - Logs a truncated query preview (to avoid logging massive SQL strings).
  - Wraps errors in `DataError` with clear codes:
    - `sql_query_error` for SQLAlchemy-specific issues.
    - `sql_load_error` for unexpected exceptions.

- `load_sql_table(table_name, schema=None, limit=None, columns=None)`  
  - Convenience wrapper for `SELECT ... FROM schema.table`.
  - Respects `database.schema` from config if explicit `schema` is not passed.
  - Allows:
    - `columns=["col1", "col2"]`
    - `limit=1000`

- `execute_sql(statement, params=None)`  
  - Executes non-SELECT statements (INSERT, UPDATE, DELETE, DDL).
  - Returns the number of affected rows (when available).
  - Wraps errors in `DataError` with code `sql_execution_error`.

Design decisions:

- **Safety first**:
  - Never log full connection strings.
  - Log only driver and small previews of queries/statements.
- **Config-driven**:
  - All connection details come from `AppConfig.database`.
- **Composable**:
  - `load_sql_query`/`load_sql_table` feed clean DataFrames into the feature layer.


### 2.3 MongoDB loading: mongodb.py

Module: `src/ml_tabular/data/mongodb.py`

This is an **optional** integration that only works if `pymongo` is installed.

Key parts:

- `_ensure_pymongo_installed()`
  - Raises `DataError("mongo_missing_dependency", ...)` if pymongo is missing.
- `_get_mongo_config()`
  - Retrieves `AppConfig.mongo` and ensures both `uri` and `database` are present.
- `_safe_mongo_log_context(uri)`
  - Uses `urllib.parse` to extract non-sensitive details:
    - scheme, hostname, port
  - Used in logs; credentials are never logged.

Main APIs:

- `get_mongo_client()`
  - Creates a `MongoClient` using config.uri.
  - Wraps connection errors in `DataError("mongo_connection_error", ...)`.

- `get_collection(collection_name)`
  - Accesses the configured database and collection.
  - Ensures any failure is wrapped with `DataError("mongo_collection_error", ...)`.

- `load_mongo_collection(collection_name, query=None, projection=None, limit=None, sort=None, drop_id=True)`
  - Runs a query on a collection and converts results to a DataFrame.
  - Options:
    - `query`: filter document selection
    - `projection`: field selection
    - `limit`: max number of documents
    - `sort`: sort order
    - `drop_id`: drop the Mongo `_id` field from the DataFrame (default True)
  - Logs size of results and query characteristics.
  - Returns an empty DataFrame if no documents match.

- `count_documents(collection_name, query=None)`
  - Counts documents matching a query.
  - Logs the operation and returns an integer.

Use cases:

- Quickly pulling semi-structured event or log data into DataFrames for feature engineering.
- Avoiding writing ad hoc scripts for one-off exports.


### 2.4 Kaggle integration: kaggle.py (planned)

Even if not fully implemented yet, the design intent is:

- A module `ml_tabular.data.kaggle` that:
  - Uses the official `kaggle` Python API (from the `mlops` extra).
  - Uses `AppConfig.kaggle` for defaults:
    - `dataset` (e.g. `"zusmani/metro-interstate-traffic-volume"`)
    - `competition` (e.g. `"titanic"`)
    - `download_subdir` (e.g. `"kaggle"` under `paths.raw_dir`)

Typical pattern:

1) Ensure you’re authenticated with Kaggle (via `~/.kaggle/kaggle.json`).
2) In code or CLI, call something like:
   - `download_dataset_if_needed(cfg.kaggle.dataset, target_dir=paths.raw_dir / cfg.kaggle.download_subdir)`
3) After the download:
   - Use `loading.load_raw_dataset` on the appropriate file (CSV / Parquet).

This lets you reproduce Kaggle-driven experiments from scratch with a single script.


## 3. Feature layer: ml_tabular.features

The feature layer is about **stateless, deterministic transformations** on DataFrames.

Why stateless?

- Easier to reason about and test.
- No hidden fit/transform state inside feature functions.
- Ideal for:
  - Logging
  - Re-applying to test data
  - Reproducibility


### 3.1 Tabular features: features/tabular.py

Module: `src/ml_tabular/features/tabular.py`

Core concept: **FeatureSpec**

- A dataclass describing what to do:

  - `datetime_columns: Sequence[str]`
    - Columns to be interpreted as datetimes and expanded into multiple features.

  - `log1p_columns: Sequence[str]`
    - Columns to transform with `log1p` (for counts/non-negative numeric).

  - `ratio_features: Mapping[str, (str, str)]`
    - `new_col = numerator / denominator`.
    - Example:
      - `"income_per_person": ("income", "household_size")`.

  - `power_features: Mapping[str, (str, float)]`
    - `new_col = source_column ** exponent`.
    - Example:
      - `"age_squared": ("age", 2.0)`.

  - `interaction_features: Mapping[str, (str, str)]`
    - `new_col = col1 * col2`.
    - Example:
      - `"rooms_x_bedrooms": ("num_rooms", "num_bedrooms")`.

  - `drop_original_datetime: bool`
    - Whether to drop original datetime columns after expansion.

There is a `DEFAULT_FEATURE_SPEC` that does nothing by default. You define your own spec either:

- In code (per project or per script).
- Indirectly via config (if you want to push feature definition into YAML later).


#### 3.1.1 build_features(df, dataset_name=None, spec=None)

Main entrypoint:

- Makes a copy of the DataFrame.
- Applies transformations in this order:
  1) datetime features (`_add_datetime_features`)
  2) log1p features (`_add_log1p_features`)
  3) ratio features (`_add_ratio_features`)
  4) power features (`_add_power_features`)
  5) interaction features (`_add_interaction_features`)

- Logs:
  - Dataset name (train/valid/test)
  - Row and column counts before and after
  - Which feature groups were applied

- Raises `DataError` if:
  - Expected columns are missing.
  - log1p columns contain negatives.
  - Non-numeric values appear in numeric-only operations (depending on context).

Each helper function:

- Works on a copy.
- Validates presence and type of columns.
- Logs or raises `DataError` with structured context.

This gives you a **clean, reusable feature function** that can be applied identically to:

- Training data
- Validation data
- Test / inference data

No hidden state, no subtle differences between splits.


### 3.2 Time-series features: features/time_series.py

Module: `src/ml_tabular/features/time_series.py`

The goal here is to bridge between **raw time-stamped data** and what **sequence models** expect.

Typical pattern:

- Start with a DataFrame like:
  - `series_id` (optional)
  - `timestamp` (datetime-like)
  - `target` (to forecast)
  - Additional covariates

- Ensure:
  - Consistent sorting by `series_id`, then `timestamp`.
  - Optional resampling or gap handling (depending on project needs).

- Add time-derived features:
  - Year, month, day, hour
  - Day-of-week, weekend flag
  - Month start/end flags
  - (Anything that helps the model pick up seasonal patterns)

This complements the PyTorch dataset in `torch/datasets/time_series.py`, which:

- Takes the **already-featured and sorted** DataFrame.
- Builds sliding windows of length `input_window`.
- Aligns targets for a horizon of `prediction_horizon`.

By putting basic time features here, you ensure:

- Consistency across experiments.
- Reusability for multiple model types (MLP over windows, RNN, Transformer, etc.).


## 4. Typical end-to-end flows


### 4.1 Tabular workflow

1) Load data:
   - From CSV:
     - `df_train = load_raw_dataset("train.csv", required_columns=[...])`
   - Or from SQL:
     - `df_train = load_sql_table("my_table", limit=100_000)`

2) Apply features:
   - Define a FeatureSpec:
     - `spec = FeatureSpec(datetime_columns=["signup_date"], log1p_columns=["num_logins"], ...)`
   - Apply:
     - `df_train_features = build_features(df_train, dataset_name="train", spec=spec)`

3) Split and feed into datasets/model:
   - Choose predictors/targets.
   - Create a `TabularDataset` with the feature DataFrame.
   - Train a `TabularMLP` via the training loop.


### 4.2 Time-series workflow

1) Load data:
   - From CSV or SQL (similar to tabular).
   - Make sure you retain:
     - series identifier (if multiple series)
     - timestamp column
     - target

2) Apply time-series features:
   - Use `ml_tabular.features.time_series` helpers to:
     - Ensure sorting by series_id/timestamp.
     - Add calendar features.

3) Build windows:
   - Use `TimeSeriesDataset` with:
     - input_window
     - prediction_horizon
     - target and covariate columns.

4) Train:
   - Use `TabularMLP` or other models over the windowed tensors.


## 5. Extending the data and feature layers

### 5.1 Adding new data sources

When you want to support a new source (e.g. S3, BigQuery, REST API):

1) Add a new module in `src/ml_tabular/data/`:
   - e.g. `s3.py`, `bigquery.py`, `api.py`

2) Extend `AppConfig` in `config.py` with a new section:
   - e.g. `S3Config`, `BigQueryConfig`, `ApiConfig`.

3) Implement functions that:
   - Use the config section for credentials/parameters.
   - Return DataFrames (or iterators of DataFrames).
   - Handle errors with `DataError` and safe logging.

4) Write tests mirroring the existing SQL/Mongo tests.


### 5.2 Adding new feature transforms

To add more feature types:

1) Extend `FeatureSpec` with additional fields:
   - e.g. `bucket_features`, `target_encoding_config`, etc.

2) Implement helper functions:
   - `_add_bucket_features`, `_add_target_encodings`, etc.

3) Wire them into `build_features` in the correct order.

4) Write unit tests that:
   - Construct a tiny DataFrame.
   - Apply the new spec.
   - Assert that the output columns and values are correct.

For **time-series-specific** features, you can:

- Add to `features/time_series.py` and keep them separate from tabular-only logic.


## 6. Summary

The **data layer** (`ml_tabular.data`) and **feature layer** (`ml_tabular.features`) work together to give you:

- **Reliable data loading** from files, SQL, MongoDB, and Kaggle.
- **Consistent, stateless feature engineering** for both tabular and time-series problems.
- A clear place to add **project-specific logic** without polluting training scripts or notebooks.

Once data and features are handled cleanly, everything else in the template (datasets, models, training loops, MLflow, CLI) can focus on **modeling and experiment management** instead of I/O and ad hoc feature code.
