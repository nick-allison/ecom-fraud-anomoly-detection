from __future__ import annotations

from typing import Any, Iterable, Mapping, Iterator

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from ml_tabular.config import get_config
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


def _safe_db_driver_name(url: str | None) -> str | None:
    """Return a safe, non-sensitive representation of the DB driver.

    For logging purposes we never want to print full URLs with credentials.
    Extract just the scheme (e.g. 'postgresql', 'sqlite', 'postgresql+psycopg').
    """
    if not url:
        return None
    return url.split("://", 1)[0]


def get_engine(*, echo: bool | None = None) -> Engine:
    """Construct and return a SQLAlchemy Engine based on AppConfig.database.

    Parameters
    ----------
    echo:
        Optional override for the SQLAlchemy 'echo' flag. If None, uses the
        value from config.database.echo.

    Raises
    ------
    DataError
        If the database configuration is missing or the engine cannot be created.
    """
    cfg = get_config()

    # Defensive: ensure `database` exists on config.
    if not hasattr(cfg, "database"):
        raise DataError(
            "Database configuration is not defined on AppConfig.",
            code="sql_missing_config",
            context={"has_database_attr": False},
            location=f"{__name__}.get_engine",
        )

    db = cfg.database
    url = getattr(db, "url", None)
    if not url:
        raise DataError(
            "Database URL is missing from configuration.",
            code="sql_missing_config",
            context={"has_database_attr": True, "url_present": False},
            location=f"{__name__}.get_engine",
        )

    driver = _safe_db_driver_name(url)
    effective_echo = db.echo if echo is None else echo

    logger.info(
        "Creating SQLAlchemy engine",
        extra={"driver": driver, "echo": effective_echo},
    )

    try:
        engine = create_engine(
            url,
            echo=effective_echo,
            future=True,
            pool_pre_ping=True,
        )
    except SQLAlchemyError as exc:
        raise DataError(
            "Failed to create SQLAlchemy engine",
            code="sql_engine_error",
            cause=exc,
            context={"driver": driver},
            location=f"{__name__}.get_engine",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error while creating SQLAlchemy engine",
            code="sql_engine_error",
            cause=exc,
            context={"driver": driver},
            location=f"{__name__}.get_engine",
        ) from exc

    return engine


def load_sql_query(
    query: str,
    *,
    params: Mapping[str, Any] | None = None,
    chunksize: int | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame | Iterator[pd.DataFrame]:
    """Execute a SQL query and return results as a DataFrame or iterator of DataFrames.

    Parameters
    ----------
    query:
        SQL query string. Use parameter placeholders appropriate for your DB
        dialect, but prefer named parameters (e.g. :start_date) where possible.
    params:
        Optional mapping of parameters to bind to the query.
    chunksize:
        If None (default), the entire result set is loaded into a single
        DataFrame. If a positive integer, returns an iterator over DataFrames
        of at most that many rows each.
    engine:
        Optional existing SQLAlchemy Engine to use. If None, `get_engine()`
        is called to construct one.

    Raises
    ------
    DataError
        If query execution fails for any reason.
    """
    engine = engine or get_engine()
    safe_query_preview = query if len(query) <= 500 else query[:497] + "..."

    logger.info(
        "Executing SQL query",
        extra={
            "query_preview": safe_query_preview,
            "has_params": bool(params),
            "chunksize": chunksize,
        },
    )

    try:
        if chunksize is None:
            df = pd.read_sql_query(text(query), engine, params=params)
            logger.info(
                "Loaded dataframe from SQL query",
                extra={
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                },
            )
            return df
        else:
            # Iterator of DataFrames
            iterator = pd.read_sql_query(
                text(query), engine, params=params, chunksize=chunksize
            )
            # Caller can iterate and log as needed.
            return iterator
    except SQLAlchemyError as exc:
        raise DataError(
            "SQLAlchemy error during query execution",
            code="sql_query_error",
            cause=exc,
            context={
                "query_preview": safe_query_preview,
                "has_params": bool(params),
                "chunksize": chunksize,
            },
            location=f"{__name__}.load_sql_query",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error during query execution",
            code="sql_load_error",
            cause=exc,
            context={
                "query_preview": safe_query_preview,
                "has_params": bool(params),
                "chunksize": chunksize,
            },
            location=f"{__name__}.load_sql_query",
        ) from exc


def load_sql_table(
    table_name: str,
    *,
    schema: str | None = None,
    limit: int | None = None,
    columns: Iterable[str] | None = None,
    engine: Engine | None = None,
) -> pd.DataFrame:
    """Load an entire table (or a subset) into a DataFrame.

    Parameters
    ----------
    table_name:
        Name of the table to load (without schema). This is expected to be
        a trusted identifier (not direct user input).
    schema:
        Optional schema name. If None, uses config.database.schema if present.
    limit:
        Optional row limit. If provided, adds a LIMIT clause to the query.
    columns:
        Optional iterable of column names to select instead of '*'. Also
        expected to be trusted identifiers.
    engine:
        Optional existing SQLAlchemy Engine to use. If None, `get_engine()`
        is used under the hood via `load_sql_query`.

    Raises
    ------
    DataError
        If the query fails.
    """
    cfg = get_config()
    db = getattr(cfg, "database", None)

    effective_schema = schema
    if effective_schema is None and db is not None:
        effective_schema = getattr(db, "schema", None)

    if columns:
        cols_expr = ", ".join(columns)
    else:
        cols_expr = "*"

    if effective_schema:
        full_name = f"{effective_schema}.{table_name}"
    else:
        full_name = table_name

    query = f"SELECT {cols_expr} FROM {full_name}"
    if limit is not None and limit > 0:
        query += f" LIMIT {int(limit)}"

    logger.info(
        "Loading table via SQL",
        extra={
            "table": table_name,
            "schema": effective_schema,
            "limit": limit,
            "columns": list(columns) if columns else None,
        },
    )

    result = load_sql_query(query, engine=engine)
    if isinstance(result, pd.DataFrame):
        return result

    # If chunksize is somehow used (not typical here), combine chunks.
    frames = list(result)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def execute_sql(
    statement: str,
    *,
    params: Mapping[str, Any] | None = None,
    engine: Engine | None = None,
) -> int:
    """Execute a non-SELECT SQL statement (INSERT, UPDATE, DELETE, DDL).

    Parameters
    ----------
    statement:
        SQL statement to execute.
    params:
        Optional mapping of parameters to bind to the statement.
    engine:
        Optional existing SQLAlchemy Engine to use. If None, `get_engine()`
        is called to construct one.

    Returns
    -------
    int
        Number of rows affected, if available. For some DDL operations this may
        be zero or undefined.

    Raises
    ------
    DataError
        If execution fails.
    """
    engine = engine or get_engine()
    safe_stmt_preview = statement if len(statement) <= 500 else statement[:497] + "..."

    logger.info(
        "Executing SQL statement",
        extra={
            "statement_preview": safe_stmt_preview,
            "has_params": bool(params),
        },
    )

    try:
        with engine.begin() as conn:
            result = conn.execute(text(statement), params or {})
            # rowcount may be -1 for some drivers/statements
            rowcount = getattr(result, "rowcount", -1)
    except SQLAlchemyError as exc:
        raise DataError(
            "SQLAlchemy error during statement execution",
            code="sql_execution_error",
            cause=exc,
            context={
                "statement_preview": safe_stmt_preview,
                "has_params": bool(params),
            },
            location=f"{__name__}.execute_sql",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error during statement execution",
            code="sql_execution_error",
            cause=exc,
            context={
                "statement_preview": safe_stmt_preview,
                "has_params": bool(params),
            },
            location=f"{__name__}.execute_sql",
        ) from exc

    logger.info(
        "Executed SQL statement",
        extra={
            "statement_preview": safe_stmt_preview,
            "rowcount": rowcount,
        },
    )
    return int(rowcount)
