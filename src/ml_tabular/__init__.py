"""
ml_tabular: opinionated template for deep-learning-friendly tabular & time-series ML.

This package exposes a small, stable public API so that projects built from this
template can do:

    from ml_tabular import (
        get_config,
        get_paths,
        AppConfig,
        TabularDataset,
        TimeSeriesSequenceDataset,
        TabularMLP,
        TabularMLPConfig,
        train_one_epoch,
        evaluate,
        fit,
        EarlyStopping,
    )

and not worry about the internal file layout.
"""

from __future__ import annotations

from importlib import metadata as _metadata

# ---------------------------------------------------------------------------
# Version
# ---------------------------------------------------------------------------

try:
    # Uses the project name from your pyproject.toml
    __version__ = _metadata.version("ml-tabular")
except _metadata.PackageNotFoundError:
    # Fallback when running from a repo without installation
    __version__ = "0.0.0"

# ---------------------------------------------------------------------------
# Public API re-exports
# ---------------------------------------------------------------------------

from .config import AppConfig, get_config, get_paths  # noqa: F401
from .exceptions import (  # noqa: F401
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
)
from .logging_config import get_logger  # noqa: F401

from .torch.datasets.tabular import TabularDataset  # noqa: F401
from .torch.datasets.time_series import TimeSeriesSequenceDataset  # noqa: F401
from .torch.models.tabular_mlp import TabularMLP, TabularMLPConfig  # noqa: F401
from .torch.training.loops import (  # noqa: F401
    EarlyStopping,
    evaluate,
    fit,
    train_one_epoch,
)

__all__ = [
    "__version__",
    # Config
    "AppConfig",
    "get_config",
    "get_paths",
    # Logging
    "get_logger",
    # Exceptions
    "ConfigError",
    "DataError",
    "ModelError",
    "TrainingError",
    # Datasets
    "TabularDataset",
    "TimeSeriesSequenceDataset",
    # Models
    "TabularMLP",
    "TabularMLPConfig",
    # Training loops
    "train_one_epoch",
    "evaluate",
    "fit",
    "EarlyStopping",
]
