from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from ml_tabular.config import AppConfig, get_config
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_cfg(cfg: Optional[AppConfig] = None) -> AppConfig:
    """Return an AppConfig, using get_config() if one is not provided."""
    return cfg or get_config()


def _import_mlflow() -> Any:
    """Import the mlflow package or raise a DataError if unavailable.

    This is only called when MLflow is actually needed (e.g., when
    cfg.mlflow.enabled is True and we attempt to start a run or log things).
    """
    try:
        import mlflow  # type: ignore
    except ImportError as exc:
        raise DataError(
            "The 'mlflow' package is not installed. "
            "Install it with `pip install mlflow` or include the 'mlops' extra "
            "from this project (e.g. `pip install -e '.[mlops]'`).",
            code="mlflow_not_installed",
            cause=exc,
            context={},
            location="ml_tabular.mlops.mlflow_utils._import_mlflow",
        ) from exc
    return mlflow


def mlflow_is_enabled(cfg: Optional[AppConfig] = None) -> bool:
    """Return True if MLflow is enabled in configuration.

    This does NOT check whether the 'mlflow' package is installed; it only
    reflects the config flag. The import will be validated lazily when
    starting a run or logging.
    """
    cfg = _get_cfg(cfg)
    return bool(cfg.mlflow.enabled)


def _flatten_dict(
    d: Mapping[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> Dict[str, Any]:
    """Flatten a nested mapping into a single-level dict with dotted keys.

    Example:
        {"training": {"epochs": 10, "lr": 0.001}}
    becomes:
        {"training.epochs": 10, "training.lr": 0.001}

    This is useful for logging structured configs as MLflow params.
    """
    items: Dict[str, Any] = {}
    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, parent_key=new_key, sep=sep))
        else:
            items[new_key] = value
    return items


# ---------------------------------------------------------------------------
# Context manager for MLflow runs
# ---------------------------------------------------------------------------


@contextmanager
def mlflow_run(
    cfg: Optional[AppConfig] = None,
    *,
    experiment_name: Optional[str] = None,
    run_name: Optional[str] = None,
    tags: Optional[Dict[str, str]] = None,
):
    """Context manager that starts/stops an MLflow run if enabled.

    Usage:

        from ml_tabular.mlops.mlflow_utils import mlflow_run, log_params, log_metrics

        with mlflow_run(cfg) as mlflow:
            if mlflow is not None:
                log_params({"model_type": "tabular_mlp"})
                log_metrics({"val_loss": 0.123}, step=1)

    Behavior:
      - If cfg.mlflow.enabled is False, this yields None and does nothing.
      - If cfg.mlflow.enabled is True:
          * imports mlflow (raising DataError if not installed),
          * sets tracking URI if cfg.mlflow.tracking_uri is configured,
          * sets the experiment (cfg.mlflow.experiment_name or cfg.experiment_name),
          * starts a run with run_name or cfg.mlflow.run_name,
          * yields the mlflow module itself for direct usage if desired.

    This design keeps your training scripts simple and lets you add/remove
    MLflow with a single config change.
    """
    cfg = _get_cfg(cfg)

    if not cfg.mlflow.enabled:
        logger.info("MLflow is disabled in configuration; skipping MLflow run.")
        yield None
        return

    mlflow = _import_mlflow()

    # Set tracking URI if configured
    if cfg.mlflow.tracking_uri:
        logger.info("Setting MLflow tracking URI to %s", cfg.mlflow.tracking_uri)
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    # Determine experiment name
    exp_name = experiment_name or cfg.mlflow.experiment_name or cfg.experiment_name
    if exp_name:
        logger.info("Setting MLflow experiment to '%s'", exp_name)
        mlflow.set_experiment(exp_name)

    # Determine run name
    effective_run_name = run_name or cfg.mlflow.run_name

    logger.info(
        "Starting MLflow run (experiment=%s, run_name=%s)",
        exp_name,
        effective_run_name,
    )

    try:
        with mlflow.start_run(run_name=effective_run_name) as run:  # noqa: F841
            # Apply tags if requested
            if tags:
                mlflow.set_tags(tags)
            yield mlflow
    except Exception as exc:
        # In case of unexpected MLflow errors, wrap in DataError for consistency.
        raise DataError(
            "An error occurred during MLflow run context.",
            code="mlflow_run_error",
            cause=exc,
            context={"experiment_name": exp_name, "run_name": effective_run_name},
            location="ml_tabular.mlops.mlflow_utils.mlflow_run",
        ) from exc
    finally:
        logger.info("MLflow run finished.")


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------


def log_params(
    params: Mapping[str, Any],
    *,
    prefix: Optional[str] = None,
    flatten: bool = False,
    cfg: Optional[AppConfig] = None,
) -> None:
    """Log parameters to the current MLflow run if enabled.

    Parameters
    ----------
    params:
        Mapping of parameter names to values.
    prefix:
        Optional prefix to prepend to parameter names (e.g. 'training.').
    flatten:
        If True, nested dicts are flattened using dotted keys via _flatten_dict.
    cfg:
        Optional AppConfig. If None, get_config() is called.

    Notes
    -----
    - If MLflow is disabled in the config, this is a no-op.
    - If MLflow is enabled but no active run exists, a warning is logged
      and nothing is logged to MLflow.
    """
    cfg = _get_cfg(cfg)
    if not cfg.mlflow.enabled:
        logger.debug("MLflow disabled; skipping log_params.")
        return

    mlflow = _import_mlflow()

    if mlflow.active_run() is None:
        logger.warning(
            "log_params called but no active MLflow run is present. "
            "Did you forget to use mlflow_run()?"
        )
        return

    data: Mapping[str, Any]
    if flatten:
        data = _flatten_dict(params)
    else:
        data = params

    # Optionally apply prefix
    to_log: Dict[str, Any] = {}
    for key, value in data.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        # Convert non-primitive values to strings as a safety measure
        if isinstance(value, (str, int, float, bool)) or value is None:
            to_log[full_key] = value
        else:
            to_log[full_key] = str(value)

    if not to_log:
        logger.debug("No parameters to log after processing; skipping.")
        return

    logger.info("Logging %d MLflow params.", len(to_log))
    mlflow.log_params(to_log)


def log_metrics(
    metrics: Mapping[str, float],
    *,
    step: Optional[int] = None,
    cfg: Optional[AppConfig] = None,
) -> None:
    """Log metrics to the current MLflow run if enabled.

    Parameters
    ----------
    metrics:
        Mapping of metric names to float values.
    step:
        Optional step index (e.g. epoch or iteration).
    cfg:
        Optional AppConfig. If None, get_config() is called.

    Notes
    -----
    - If MLflow is disabled, this is a no-op.
    - If no active run exists, a warning is logged and metrics are not logged.
    """
    cfg = _get_cfg(cfg)
    if not cfg.mlflow.enabled:
        logger.debug("MLflow disabled; skipping log_metrics.")
        return

    mlflow = _import_mlflow()

    if mlflow.active_run() is None:
        logger.warning(
            "log_metrics called but no active MLflow run is present. "
            "Did you forget to use mlflow_run()?"
        )
        return

    if not metrics:
        logger.debug("Empty metrics mapping provided; skipping log_metrics.")
        return

    logger.info("Logging %d MLflow metrics (step=%s).", len(metrics), step)
    mlflow.log_metrics(metrics, step=step)


def log_artifacts_from_dir(
    dir_path: Path | str,
    *,
    artifact_path: Optional[str] = None,
    cfg: Optional[AppConfig] = None,
) -> None:
    """Log all files in a directory as MLflow artifacts if enabled.

    Parameters
    ----------
    dir_path:
        Directory whose contents should be logged as artifacts.
    artifact_path:
        Optional subdirectory within the MLflow run's artifact store.
    cfg:
        Optional AppConfig. If None, get_config() is called.

    Notes
    -----
    - If MLflow is disabled, this is a no-op.
    - If no active run exists, a warning is logged and artifacts are not logged.
    """
    cfg = _get_cfg(cfg)
    if not cfg.mlflow.enabled:
        logger.debug("MLflow disabled; skipping log_artifacts_from_dir.")
        return

    mlflow = _import_mlflow()

    if mlflow.active_run() is None:
        logger.warning(
            "log_artifacts_from_dir called but no active MLflow run is present. "
            "Did you forget to use mlflow_run()?"
        )
        return

    path = Path(dir_path).resolve()
    if not path.exists() or not path.is_dir():
        raise DataError(
            f"Artifact directory does not exist or is not a directory: {path}",
            code="mlflow_artifact_dir_invalid",
            context={"dir_path": str(path)},
            location="ml_tabular.mlops.mlflow_utils.log_artifacts_from_dir",
        )

    logger.info(
        "Logging artifacts from directory %s (artifact_path=%s).",
        path,
        artifact_path,
    )
    mlflow.log_artifacts(str(path), artifact_path=artifact_path)
