"""
Command-line interface for the ml_tabular template.

This CLI is intentionally thin: it delegates actual training logic to the
root-level scripts (train_tabular_mlp.py, train_ts_mlp.py), so that the
heavy lifting lives in a single place and stays easy to reason about.

Typical usage (after installing the package or from the repo root):

    ml-tabular train-tabular --config configs/tabular/train_tabular_baseline.yaml
    ml-tabular train-ts       --config configs/time_series/train_ts_baseline.yaml

The console script entry point in pyproject.toml should look like:

    [project.scripts]
    ml-tabular = "ml_tabular.cli:app"

so that `ml-tabular` is available on the command line.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .logging_config import get_logger

app = typer.Typer(
    help="CLI for the ml_tabular template (tabular & time-series ML).",
    no_args_is_help=True,
)

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Return the project root directory (one level above 'src').

    This assumes the layout:

        <repo_root>/
          src/
            ml_tabular/
              cli.py
          train_tabular_mlp.py
          train_ts_mlp.py
          configs/
            ...

    which matches the template we've been building.
    """
    return Path(__file__).resolve().parents[2]


def _run_script(
    script_name: str,
    *,
    config: Path,
    env: Optional[str],
    device: str,
    no_mlflow: bool,
) -> None:
    """Execute a root-level training script with the provided arguments.

    Parameters
    ----------
    script_name:
        Name of the Python script at the project root, e.g. "train_tabular_mlp.py".
    config:
        Path to the YAML config file.
    env:
        Optional environment/profile name (e.g. "dev", "prod").
    device:
        Device string: "auto", "cpu", or "cuda".
    no_mlflow:
        If True, pass --no-mlflow to the script.

    Raises
    ------
    SystemExit:
        If the underlying script returns a non-zero exit code, this function
        raises SystemExit with that code.
    """
    root = _project_root()
    script_path = root / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Expected script '{script_name}' not found at {script_path}. "
            "Make sure you're running this from a clone of the full template repo, "
            "and that the training scripts are present at the project root."
        )

    cmd = [
        sys.executable,
        str(script_path),
        "--config",
        str(config),
    ]

    if env is not None:
        cmd.extend(["--env", env])

    if device:
        cmd.extend(["--device", device])

    if no_mlflow:
        cmd.append("--no-mlflow")

    logger.info("Running command: %s", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        # Propagate the exit code so the CLI behaves like the underlying script.
        logger.error("Training script '%s' failed with exit code %d", script_name, exc.returncode)
        raise SystemExit(exc.returncode) from exc


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("version")
def version() -> None:
    """Print the installed ml_tabular version."""
    typer.echo(f"ml_tabular version: {__version__}")


@app.command("train-tabular")
def train_tabular(
    config: Path = typer.Option(
        Path("configs/tabular/train_tabular_baseline.yaml"),
        "--config",
        "-c",
        exists=False,
        dir_okay=False,
        readable=True,
        help=(
            "Path to the YAML config file for tabular training. "
            "Defaults to configs/tabular/train_tabular_baseline.yaml."
        ),
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help=(
            "Config environment/profile name (e.g. 'dev', 'prod'). "
            "If omitted, the default from ml_tabular.config is used."
        ),
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        case_sensitive=False,
        help="Device to use: 'auto' (CUDA if available, else CPU), 'cpu', or 'cuda'.",
    ),
    no_mlflow: bool = typer.Option(
        False,
        "--no-mlflow",
        help="Disable MLflow logging even if configured.",
    ),
) -> None:
    """Train a tabular MLP using the train_tabular_mlp.py script."""
    _run_script(
        "train_tabular_mlp.py",
        config=config,
        env=env,
        device=device,
        no_mlflow=no_mlflow,
    )


@app.command("train-ts")
def train_time_series(
    config: Path = typer.Option(
        Path("configs/time_series/train_ts_baseline.yaml"),
        "--config",
        "-c",
        exists=False,
        dir_okay=False,
        readable=True,
        help=(
            "Path to the YAML config file for time-series training. "
            "Defaults to configs/time_series/train_ts_baseline.yaml."
        ),
    ),
    env: Optional[str] = typer.Option(
        None,
        "--env",
        help=(
            "Config environment/profile name (e.g. 'dev', 'prod'). "
            "If omitted, the default from ml_tabular.config is used."
        ),
    ),
    device: str = typer.Option(
        "auto",
        "--device",
        case_sensitive=False,
        help="Device to use: 'auto' (CUDA if available, else CPU), 'cpu', or 'cuda'.",
    ),
    no_mlflow: bool = typer.Option(
        False,
        "--no-mlflow",
        help="Disable MLflow logging even if configured.",
    ),
) -> None:
    """Train a time-series MLP using the train_ts_mlp.py script."""
    _run_script(
        "train_ts_mlp.py",
        config=config,
        env=env,
        device=device,
        no_mlflow=no_mlflow,
    )


# ---------------------------------------------------------------------------
# Entry point for `python -m ml_tabular.cli`
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app()
