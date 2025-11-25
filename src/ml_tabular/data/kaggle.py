from __future__ import annotations

from pathlib import Path
from typing import Optional

from ml_tabular.config import AppConfig, get_config
from ml_tabular.exceptions import DataError
from ml_tabular.logging_config import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _import_kaggle_api():
    """Import and return the KaggleApi class, or raise a DataError if unavailable."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
    except ImportError as exc:
        raise DataError(
            "The 'kaggle' package is not installed. "
            "Install it with `pip install kaggle` or include the 'mlops' extra "
            "from this project (e.g. `pip install -e '.[mlops]'`).",
            code="kaggle_not_installed",
            cause=exc,
            context={},
            location="ml_tabular.data.kaggle._import_kaggle_api",
        ) from exc
    return KaggleApi


def _authenticate_kaggle(api) -> None:
    """Authenticate the Kaggle API, wrapping errors in a DataError.

    This uses Kaggle's standard credential mechanisms:
      - ~/.kaggle/kaggle.json
      - KAGGLE_USERNAME / KAGGLE_KEY environment variables

    If credentials are missing or invalid, a DataError is raised with
    guidance on how to set them up.
    """
    try:
        api.authenticate()
    except Exception as exc:  # KaggleApi doesn't expose a specific exception type
        raise DataError(
            "Failed to authenticate with Kaggle API. "
            "Ensure that you have valid credentials configured either in "
            "~/.kaggle/kaggle.json or via KAGGLE_USERNAME / KAGGLE_KEY.",
            code="kaggle_auth_failed",
            cause=exc,
            context={},
            location="ml_tabular.data.kaggle._authenticate_kaggle",
        ) from exc


def _get_default_download_dir(cfg: Optional[AppConfig] = None) -> Path:
    """Return the default directory for Kaggle downloads.

    This is resolved as:
        resolved_paths.raw_dir / cfg.kaggle.download_subdir

    Example (with defaults):
        data/raw/kaggle
    """
    if cfg is None:
        cfg = get_config()
    paths = cfg.resolved_paths()
    subdir = cfg.kaggle.download_subdir or "kaggle"
    download_dir = (paths.raw_dir / subdir).resolve()
    download_dir.mkdir(parents=True, exist_ok=True)
    return download_dir


def _ensure_directory(path: Path) -> Path:
    """Ensure that `path` exists and is a directory, creating it if needed."""
    if path.exists() and not path.is_dir():
        raise DataError(
            f"Download path exists but is not a directory: {path}",
            code="kaggle_path_not_dir",
            context={"path": str(path)},
            location="ml_tabular.data.kaggle._ensure_directory",
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_dataset(
    dataset: Optional[str] = None,
    dest: Optional[Path | str] = None,
    *,
    unzip: bool = True,
    force: bool = False,
    cfg: Optional[AppConfig] = None,
) -> Path:
    """Download a Kaggle dataset into the project's raw data directory.

    Parameters
    ----------
    dataset:
        Kaggle dataset identifier, e.g. "zusmani/metro-interstate-traffic-volume".
        If None, this function will fall back to cfg.kaggle.dataset. If that is
        also None, a DataError is raised.
    dest:
        Optional destination directory. If None, the directory is resolved as:
            resolved_paths.raw_dir / cfg.kaggle.download_subdir / <dataset_slug>
        where <dataset_slug> is derived from the dataset identifier (text after '/').
    unzip:
        Whether to unzip the downloaded archive. Defaults to True.
    force:
        If True, forces download even if the file/directory already exists,
        by passing force=True to the Kaggle API.
    cfg:
        Optional AppConfig instance. If None, get_config() is called.

    Returns
    -------
    Path
        The directory into which the dataset was downloaded.

    Raises
    ------
    DataError
        If the Kaggle package is not installed, authentication fails, the dataset
        identifier is missing, or the download itself fails.
    """
    if cfg is None:
        cfg = get_config()

    dataset_id = dataset or cfg.kaggle.dataset
    if not dataset_id:
        raise DataError(
            "No Kaggle dataset specified. Provide a 'dataset' argument or set "
            "kaggle.dataset in the YAML configuration.",
            code="kaggle_dataset_missing",
            context={},
            location="ml_tabular.data.kaggle.download_dataset",
        )

    # Determine destination directory
    if dest is not None:
        dest_path = Path(dest).resolve()
    else:
        base_dir = _get_default_download_dir(cfg)
        # Try to derive a simple slug from the dataset identifier, e.g. "user/dataset"
        # -> "dataset"
        slug = dataset_id.split("/")[-1]
        dest_path = (base_dir / slug).resolve()

    _ensure_directory(dest_path)

    KaggleApi = _import_kaggle_api()
    api = KaggleApi()
    _authenticate_kaggle(api)

    logger.info("Downloading Kaggle dataset '%s' to %s (unzip=%s, force=%s)",
                dataset_id, dest_path, unzip, force)

    try:
        # The Kaggle API writes files into the provided path.
        api.dataset_download_files(
            dataset=dataset_id,
            path=str(dest_path),
            unzip=unzip,
            quiet=False,
            force=force,
        )
    except Exception as exc:
        raise DataError(
            f"Failed to download Kaggle dataset '{dataset_id}'.",
            code="kaggle_dataset_download_failed",
            cause=exc,
            context={"dataset": dataset_id, "dest": str(dest_path)},
            location="ml_tabular.data.kaggle.download_dataset",
        ) from exc

    logger.info("Finished downloading dataset '%s' into %s", dataset_id, dest_path)
    return dest_path


def download_competition(
    competition: Optional[str] = None,
    dest: Optional[Path | str] = None,
    *,
    unzip: bool = True,
    force: bool = False,
    cfg: Optional[AppConfig] = None,
) -> Path:
    """Download a Kaggle competition dataset into the project's raw data directory.

    Parameters
    ----------
    competition:
        Kaggle competition identifier, e.g. "titanic".
        If None, this function will fall back to cfg.kaggle.competition. If that is
        also None, a DataError is raised.
    dest:
        Optional destination directory. If None, the directory is resolved as:
            resolved_paths.raw_dir / cfg.kaggle.download_subdir / <competition>
    unzip:
        Whether to unzip the downloaded archive. Defaults to True.
    force:
        If True, forces download even if the file/directory already exists,
        by passing force=True to the Kaggle API.
    cfg:
        Optional AppConfig instance. If None, get_config() is called.

    Returns
    -------
    Path
        The directory into which the competition files were downloaded.

    Raises
    ------
    DataError
        If the Kaggle package is not installed, authentication fails, the competition
        identifier is missing, or the download itself fails.
    """
    if cfg is None:
        cfg = get_config()

    competition_id = competition or cfg.kaggle.competition
    if not competition_id:
        raise DataError(
            "No Kaggle competition specified. Provide a 'competition' argument or set "
            "kaggle.competition in the YAML configuration.",
            code="kaggle_competition_missing",
            context={},
            location="ml_tabular.data.kaggle.download_competition",
        )

    # Determine destination directory
    if dest is not None:
        dest_path = Path(dest).resolve()
    else:
        base_dir = _get_default_download_dir(cfg)
        dest_path = (base_dir / competition_id).resolve()

    _ensure_directory(dest_path)

    KaggleApi = _import_kaggle_api()
    api = KaggleApi()
    _authenticate_kaggle(api)

    logger.info("Downloading Kaggle competition '%s' to %s (unzip=%s, force=%s)",
                competition_id, dest_path, unzip, force)

    try:
        api.competition_download_files(
            competition=competition_id,
            path=str(dest_path),
            unzip=unzip,
            quiet=False,
            force=force,
        )
    except Exception as exc:
        raise DataError(
            f"Failed to download Kaggle competition '{competition_id}'.",
            code="kaggle_competition_download_failed",
            cause=exc,
            context={"competition": competition_id, "dest": str(dest_path)},
            location="ml_tabular.data.kaggle.download_competition",
        ) from exc

    logger.info("Finished downloading competition '%s' into %s", competition_id, dest_path)
    return dest_path
