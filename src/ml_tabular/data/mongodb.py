from __future__ import annotations

from typing import Any, Mapping, Sequence

import pandas as pd

from ml_tabular.config import get_config
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)

try:  # Optional dependency: pymongo
    from pymongo import MongoClient
    from pymongo.collection import Collection
    from pymongo.errors import PyMongoError
except ImportError:  # pragma: no cover - handled at runtime in guards below
    MongoClient = None  # type: ignore[assignment]
    Collection = Any  # type: ignore[assignment]

    class PyMongoError(Exception):  # type: ignore[no-redef]
        """Fallback error type used when pymongo is not installed."""
        pass


def _ensure_pymongo_installed() -> None:
    """Raise a DataError if pymongo is not installed."""
    if MongoClient is None:
        raise DataError(
            "pymongo is not installed. Install it with 'pip install pymongo' "
            "or add 'pymongo' to your project dependencies.",
            code="mongo_missing_dependency",
            context={},
            location=f"{__name__}._ensure_pymongo_installed",
        )


def _get_mongo_config():
    """Return the MongoConfig from AppConfig or raise DataError if missing."""
    cfg = get_config()
    mongo_cfg = getattr(cfg, "mongo", None)
    if mongo_cfg is None:
        raise DataError(
            "MongoDB configuration is not defined on AppConfig.",
            code="mongo_missing_config",
            context={"has_mongo_attr": False},
            location=f"{__name__}._get_mongo_config",
        )

    uri = getattr(mongo_cfg, "uri", None)
    database = getattr(mongo_cfg, "database", None)
    if not uri or not database:
        raise DataError(
            "MongoDB configuration must define both 'uri' and 'database'.",
            code="mongo_missing_config",
            context={"has_mongo_attr": True, "has_uri": bool(uri), "has_database": bool(database)},
            location=f"{__name__}._get_mongo_config",
        )
    return mongo_cfg


def _safe_mongo_log_context(uri: str) -> dict[str, Any]:
    """Return a non-sensitive representation of the MongoDB connection.

    We never log full URIs because they may contain credentials.
    Instead, we extract scheme and host information only.
    """
    from urllib.parse import urlparse

    parsed = urlparse(uri)
    return {
        "scheme": parsed.scheme or "mongodb",
        "host": parsed.hostname,
        "port": parsed.port,
    }


def get_mongo_client() -> MongoClient:
    """Construct and return a MongoClient based on AppConfig.mongo.

    Raises
    ------
    DataError
        If pymongo is missing or the configuration/connection fails.
    """
    _ensure_pymongo_installed()
    mongo_cfg = _get_mongo_config()
    uri = mongo_cfg.uri

    log_ctx = _safe_mongo_log_context(uri)
    logger.info(
        "Creating MongoDB client",
        extra={"mongo": log_ctx},
    )

    try:
        client = MongoClient(uri)  # type: ignore[call-arg]
    except PyMongoError as exc:  # type: ignore[misc]
        raise DataError(
            "Failed to connect to MongoDB",
            code="mongo_connection_error",
            cause=exc,
            context=log_ctx,
            location=f"{__name__}.get_mongo_client",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error while creating MongoDB client",
            code="mongo_connection_error",
            cause=exc,
            context=log_ctx,
            location=f"{__name__}.get_mongo_client",
        ) from exc

    return client  # type: ignore[return-value]


def get_collection(
    collection_name: str,
    *,
    client: MongoClient | None = None,
) -> Collection:
    """Return a pymongo Collection using AppConfig.mongo.database.

    Parameters
    ----------
    collection_name:
        Name of the MongoDB collection.
    client:
        Optional existing MongoClient instance. If None, `get_mongo_client()`
        is used to construct one.

    Raises
    ------
    DataError
        If configuration is invalid or client creation fails.
    """
    _ensure_pymongo_installed()
    mongo_cfg = _get_mongo_config()
    client = client or get_mongo_client()

    db_name = mongo_cfg.database
    logger.info(
        "Accessing MongoDB collection",
        extra={"database": db_name, "collection": collection_name},
    )

    try:
        db = client[db_name]
        collection = db[collection_name]
    except Exception as exc:
        raise DataError(
            "Failed to access MongoDB collection",
            code="mongo_collection_error",
            cause=exc,
            context={"database": db_name, "collection": collection_name},
            location=f"{__name__}.get_collection",
        ) from exc

    return collection  # type: ignore[return-value]


def load_mongo_collection(
    collection_name: str,
    *,
    query: Mapping[str, Any] | None = None,
    projection: Mapping[str, Any] | None = None,
    limit: int | None = None,
    sort: Sequence[tuple[str, int]] | None = None,
    drop_id: bool = True,
    client: MongoClient | None = None,
) -> pd.DataFrame:
    """Load documents from a MongoDB collection into a pandas DataFrame.

    Parameters
    ----------
    collection_name:
        Name of the MongoDB collection to query.
    query:
        MongoDB filter query (e.g. {"status": "active"}). Defaults to {}.
    projection:
        Projection dict to select specific fields (e.g. {"_id": 0, "field": 1}).
        If None, all fields are returned.
    limit:
        Optional maximum number of documents to retrieve.
    sort:
        Optional sort specification as a sequence of (field, direction) tuples,
        e.g. [("created_at", -1)] for descending.
    drop_id:
        If True (default), drop the '_id' column from the resulting DataFrame
        if present.
    client:
        Optional existing MongoClient instance to use. If None, a client is
        created via `get_mongo_client()`.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the documents.

    Raises
    ------
    DataError
        If query execution fails or pymongo is not installed.
    """
    _ensure_pymongo_installed()
    collection = get_collection(collection_name, client=client)
    effective_query: Mapping[str, Any] = query or {}

    logger.info(
        "Querying MongoDB collection",
        extra={
            "collection": collection_name,
            "has_query": bool(query),
            "has_projection": bool(projection),
            "limit": limit,
            "sort": sort,
        },
    )

    try:
        cursor = collection.find(effective_query, projection)

        if sort:
            cursor = cursor.sort(sort)

        if limit is not None and limit > 0:
            cursor = cursor.limit(limit)

        docs = list(cursor)
    except PyMongoError as exc:  # type: ignore[misc]
        raise DataError(
            "Failed to load documents from MongoDB",
            code="mongo_load_error",
            cause=exc,
            context={
                "collection": collection_name,
                "has_query": bool(query),
                "limit": limit,
                "sort": sort,
            },
            location=f"{__name__}.load_mongo_collection",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error while loading documents from MongoDB",
            code="mongo_load_error",
            cause=exc,
            context={
                "collection": collection_name,
                "has_query": bool(query),
                "limit": limit,
                "sort": sort,
            },
            location=f"{__name__}.load_mongo_collection",
        ) from exc

    if not docs:
        logger.info(
            "MongoDB query returned no documents",
            extra={"collection": collection_name},
        )
        return pd.DataFrame()

    df = pd.DataFrame(docs)

    if drop_id and "_id" in df.columns:
        df = df.drop(columns=["_id"])

    logger.info(
        "Loaded dataframe from MongoDB collection",
        extra={
            "collection": collection_name,
            "n_rows": int(df.shape[0]),
            "n_cols": int(df.shape[1]),
        },
    )

    return df


def count_documents(
    collection_name: str,
    *,
    query: Mapping[str, Any] | None = None,
    client: MongoClient | None = None,
) -> int:
    """Count documents in a collection matching the given query.

    Parameters
    ----------
    collection_name:
        Name of the MongoDB collection.
    query:
        Filter query. Defaults to {} (count all documents).
    client:
        Optional existing MongoClient instance to use. If None, `get_mongo_client()`
        is used behind the scenes.

    Returns
    -------
    int
        Number of matching documents.

    Raises
    ------
    DataError
        If the count operation fails or pymongo is not installed.
    """
    _ensure_pymongo_installed()
    collection = get_collection(collection_name, client=client)
    effective_query: Mapping[str, Any] = query or {}

    logger.info(
        "Counting documents in MongoDB collection",
        extra={"collection": collection_name, "has_query": bool(query)},
    )

    try:
        count = collection.count_documents(effective_query)
    except PyMongoError as exc:  # type: ignore[misc]
        raise DataError(
            "Failed to count documents in MongoDB",
            code="mongo_count_error",
            cause=exc,
            context={"collection": collection_name, "has_query": bool(query)},
            location=f"{__name__}.count_documents",
        ) from exc
    except Exception as exc:
        raise DataError(
            "Unexpected error while counting documents in MongoDB",
            code="mongo_count_error",
            cause=exc,
            context={"collection": collection_name, "has_query": bool(query)},
            location=f"{__name__}.count_documents",
        ) from exc

    logger.info(
        "Counted documents in MongoDB collection",
        extra={"collection": collection_name, "count": count},
    )
    return int(count)
