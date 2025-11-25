from __future__ import annotations

import logging
import logging.config
import os
from pathlib import Path
from typing import Any, Mapping

# ----------------------------------------------------------------------
# Environment-driven defaults
# ----------------------------------------------------------------------

# Prefer project-specific env vars; fall back to generic ones where sensible.
DEFAULT_LOG_LEVEL = (
    os.getenv("ML_TABULAR_LOG_LEVEL")
    or os.getenv("LOG_LEVEL", "INFO")
).upper()

DEFAULT_LOG_DIR = Path(
    os.getenv("ML_TABULAR_LOG_DIR") or os.getenv("LOG_DIR", "logs")
)

# Environment name used primarily to decide handler behaviour (console-only vs file+console).
APP_ENV = (
    os.getenv("ML_TABULAR_ENV")  # aligned with config.AppConfig ENV_PREFIX
    or os.getenv("ENV")
    or "dev"
).lower()

# Control auto-configuration behaviour for get_logger().
AUTO_CONFIG = os.getenv("ML_TABULAR_CONFIGURE_LOGGING", "1").lower() not in {
    "0",
    "false",
    "no",
}

# Log format: "text" (default) or "json".
LOG_FORMAT = os.getenv("ML_TABULAR_LOG_FORMAT", "text").lower()

_LOG_CONFIGURED = False


def _supports_json_logging() -> bool:
    """Return True if a JSON formatter is available."""
    try:
        import pythonjsonlogger  # type: ignore[unused-import]
    except ImportError:
        return False
    return True


def _build_formatters(fmt: str) -> dict[str, Any]:
    """Build formatter configuration based on the requested format."""
    if fmt == "json" and _supports_json_logging():
        # JSON logging: requires `python-json-logger` to be installed.
        # pip install python-json-logger
        return {
            "json": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(levelname)s %(name)s %(message)s",
            },
            "json_verbose": {
                "class": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": (
                    "%(asctime)s %(levelname)s %(name)s "
                    "%(filename)s %(lineno)d %(message)s"
                ),
            },
        }

    # Fallback / default: human-friendly text formatters.
    return {
        "standard": {
            "format": "[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        },
        "verbose": {
            "format": (
                "[%(asctime)s] [%(levelname)s] %(name)s "
                "(%(filename)s:%(lineno)d) - %(message)s"
            ),
        },
    }


def _build_logging_config(
    env: str,
    log_dir: Path,
    level: str,
    fmt: str,
) -> dict[str, Any]:
    """Return a dictConfig-style logging configuration.

    This sets up:
    - A console handler (always).
    - A rotating file handler (typically used in 'prod').

    The active handlers for root / ml_tabular loggers depend on `env`.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "app.log"

    formatters = _build_formatters(fmt)

    # Choose formatter keys depending on whether we're in JSON or text mode.
    if "json" in formatters:
        console_formatter = "json"
        file_formatter = "json_verbose"
    else:
        console_formatter = "standard"
        file_formatter = "verbose"

    # Decide which handlers the root and package loggers should use
    # based on environment.
    env = env.lower()
    if env in {"prod", "production"}:
        root_handlers = ["console", "file"]
        ml_tabular_handlers = ["console", "file"]
    else:
        # dev, test, anything else -> console only by default
        root_handlers = ["console"]
        ml_tabular_handlers = ["console"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": formatters,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": console_formatter,
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": level,
                "formatter": file_formatter,
                "filename": str(log_file),
                "maxBytes": 10 * 1024 * 1024,  # 10 MB
                "backupCount": 5,
                "encoding": "utf-8",
            },
        },
        # Root logger: catches everything by default.
        "root": {
            "level": level,
            "handlers": root_handlers,
        },
        # Package-specific logger: `ml_tabular.*`
        "loggers": {
            "ml_tabular": {
                "level": level,
                "handlers": ml_tabular_handlers,
                "propagate": False,
            },
        },
    }


def configure_logging(
    *,
    level: str | None = None,
    log_dir: Path | str | None = None,
    env: str | None = None,
    fmt: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    force: bool = False,
) -> None:
    """Configure the global logging system for the application.

    Typically called once at process startup (e.g. in your CLI entrypoint).

    Parameters
    ----------
    level:
        Log level ("DEBUG", "INFO", "WARNING", etc.). Defaults to env or "INFO".
    log_dir:
        Directory where log files are written. Defaults to env or "logs".
    env:
        Application environment ("dev", "prod", "test"). Defaults to ML_TABULAR_ENV/ENV.
    fmt:
        Log format: "text" (default) or "json". Defaults to ML_TABULAR_LOG_FORMAT.
    extra_config:
        Optional dictConfig-style overrides merged (shallowly) into the base config.
    force:
        If True, re-configure logging even if it was already configured.
    """
    global _LOG_CONFIGURED

    if _LOG_CONFIGURED and not force:
        return

    effective_level = (level or DEFAULT_LOG_LEVEL).upper()
    effective_dir = Path(log_dir) if log_dir is not None else DEFAULT_LOG_DIR
    effective_env = (env or APP_ENV).lower()
    effective_fmt = (fmt or LOG_FORMAT).lower()

    if effective_fmt == "json" and not _supports_json_logging():
        # Fall back to text if JSON is requested but not available.
        logging.getLogger(__name__).warning(
            "JSON logging requested but python-json-logger is not installed; "
            "falling back to text format."
        )
        effective_fmt = "text"

    config = _build_logging_config(
        env=effective_env,
        log_dir=effective_dir,
        level=effective_level,
        fmt=effective_fmt,
    )

    if extra_config:
        # Shallow merge for simple overrides.
        for key, value in extra_config.items():
            if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                config[key].update(value)
            else:
                config[key] = value

    logging.config.dictConfig(config)
    _LOG_CONFIGURED = True


def configure_logging_from_app_config(
    app_config: Any,
    *,
    fmt: str | None = None,
    extra_config: Mapping[str, Any] | None = None,
    force: bool = False,
) -> None:
    """Convenience helper to configure logging from an AppConfig instance.

    This is intended to work with `ml_tabular.config.AppConfig`, but is typed
    as `Any` to avoid import cycles and keep this module lightweight.

        from ml_tabular.config import get_config
        from ml_tabular.logging_config import configure_logging_from_app_config

        cfg = get_config()
        configure_logging_from_app_config(cfg)
    """
    # We assume the config object exposes `.env`, `.log_level`, and `.resolved_paths()`.
    env = getattr(app_config, "env", "dev")
    level = getattr(app_config, "log_level", "INFO")
    paths = app_config.paths if hasattr(app_config, "paths") else None

    if paths is not None and hasattr(paths, "base_dir"):
        base_dir = Path(paths.base_dir)
        log_dir = base_dir / "logs"
    else:
        log_dir = DEFAULT_LOG_DIR

    configure_logging(
        level=str(level),
        log_dir=log_dir,
        env=str(env),
        fmt=fmt,
        extra_config=extra_config,
        force=force,
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a logger, ensuring that logging is configured.

    If ML_TABULAR_CONFIGURE_LOGGING=0/false/no, this will *not* auto-configure
    logging and will simply return `logging.getLogger(name)` using whatever
    configuration the host application has set up.

        from ml_tabular.logging_config import get_logger

        logger = get_logger(__name__)
        logger.info("Hello from module")
    """
    if not _LOG_CONFIGURED and AUTO_CONFIG:
        configure_logging()

    return logging.getLogger(name)
