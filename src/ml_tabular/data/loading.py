from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ml_tabular.config import get_paths, PathsConfig
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


def _infer_format(path: Path) -> str:
    """Infer file format from suffix, defaulting to 'csv'."""
    suffix = path.suffix.lower()
    if suffix in {".csv"}:
        return "csv"
    if suffix in {".parquet"}:
        return "parquet"
    if suffix in {".feather"}:
        return "feather"
    # Default to CSV if we don't recognize the extension.
    return "csv"


def _ensure_exists(path: Path) -> None:
    """Raise DataError if path does not exist."""
    if not path.exists():
        raise DataError(
            f"Data file not found: {path}",
            code="data_file_not_found",
            context={"path": str(path)},
            location=f"{__name__}._ensure_exists",
        )


def _validate_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str] | None = None,
    *,
    path: Path | None = None,
) -> None:
    """Validate that required columns are present in the dataframe.

    Raises
    ------
    DataError
        If any required columns are missing.
    """
    if not required_columns:
        return

    required_set = set(required_columns)
    missing = required_set.difference(df.columns)
    if missing:
        raise DataError(
            "Missing required columns in loaded dataset",
            code="data_missing_columns",
            context={
                "path": str(path) if path is not None else None,
                "missing_columns": sorted(missing),
                "available_columns": list(df.columns),
            },
            location=f"{__name__}._validate_columns",
        )


def load_dataframe(
    path: str | Path,
    *,
    format: str | None = None,
    required_columns: Iterable[str] | None = None,
    read_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Load a tabular dataset into a pandas DataFrame with robust error handling.

    Parameters
    ----------
    path:
        Path to the file to load.
    format:
        Optional format override: 'csv', 'parquet', or 'feather'.
        If omitted, inferred from the file suffix.
    required_columns:
        Optional iterable of column names that must be present in the dataset.
        If any are missing, a DataError is raised.
    read_kwargs:
        Optional additional keyword arguments forwarded to the pandas reader
        (e.g. {"sep": ";", "dtype": {"col": "category"}}).

    Raises
    ------
    DataError
        If the file does not exist, cannot be read, or fails column validation.
    """
    path = Path(path)
    _ensure_exists(path)

    fmt = (format or _infer_format(path)).lower()
    kwargs: dict[str, Any] = dict(read_kwargs or {})

    logger.info(
        "Loading dataframe",
        extra={"path": str(path), "format": fmt, "read_kwargs": kwargs},
    )

    try:
        if fmt == "csv":
            df = pd.read_csv(path, **kwargs)
        elif fmt == "parquet":
            df = pd.read_parquet(path, **kwargs)
        elif fmt == "feather":
            df = pd.read_feather(path, **kwargs)
        else:
            raise DataError(
                f"Unsupported data format: {fmt}",
                code="data_unsupported_format",
                context={"path": str(path), "format": fmt},
                location=f"{__name__}.load_dataframe",
            )
    except DataError:
        # Already a structured DataError, just bubble it up.
        raise
    except Exception as exc:
        # Wrap any other exception in a DataError for consistent handling.
        raise DataError(
            f"Failed to load dataframe from {path}",
            code="data_load_error",
            cause=exc,
            context={"path": str(path), "format": fmt},
            location=f"{__name__}.load_dataframe",
        ) from exc

    _validate_columns(df, required_columns, path=path)

    logger.info(
        "Loaded dataframe",
        extra={
            "path": str(path),
            "format": fmt,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
        },
    )

    return df


def load_raw_dataset(
    filename: str,
    *,
    paths: PathsConfig | None = None,
    required_columns: Iterable[str] | None = None,
    read_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Convenience helper to load a dataset from the `raw_dir`."""
    if paths is None:
        paths = get_paths()

    path = paths.raw_dir / filename
    return load_dataframe(
        path,
        required_columns=required_columns,
        read_kwargs=read_kwargs,
    )


def load_processed_dataset(
    filename: str,
    *,
    paths: PathsConfig | None = None,
    required_columns: Iterable[str] | None = None,
    read_kwargs: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Convenience helper to load a dataset from the `processed_dir`."""
    if paths is None:
        paths = get_paths()

    path = paths.processed_dir / filename
    return load_dataframe(
        path,
        required_columns=required_columns,
        read_kwargs=read_kwargs,
    )
