from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from ml_tabular.cli import app
from ml_tabular.config import get_paths


runner = CliRunner()


# ---------------------------------------------------------------------------
# Root CLI behaviour
# ---------------------------------------------------------------------------


def test_cli_root_help_shows_commands() -> None:
    """`ml-tabular --help` (via app) should render help and list key commands."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0, result.stdout

    stdout = result.stdout
    # We expect train-tabular and some time-series training command
    assert "train-tabular" in stdout
    assert ("train-ts" in stdout) or ("train-time-series" in stdout)


# ---------------------------------------------------------------------------
# train-tabular command
# ---------------------------------------------------------------------------


def test_cli_train_tabular_smoke(
    tabular_config_path: Path,
) -> None:
    """train-tabular should run end-to-end and produce at least one model artifact.

    Uses the temporary config from conftest.py and disables MLflow integration
    so the test does not require mlflow to be installed.
    """
    # Ensure paths/models exist and are empty at the start
    paths = get_paths(config_path=tabular_config_path, force_reload=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    for f in paths.models_dir.iterdir():
        f.unlink()

    result = runner.invoke(
        app,
        [
            "train-tabular",
            "--config",
            str(tabular_config_path),
            "--device",
            "cpu",
            "--no-mlflow",
        ],
    )

    assert result.exit_code == 0, result.stdout

    # After running, we expect at least one model-like artifact in models_dir
    model_files = list(paths.models_dir.glob("*.pt")) + list(
        paths.models_dir.glob("*.pth")
    )
    assert (
        len(model_files) >= 1
    ), f"Expected at least one .pt/.pth file in {paths.models_dir}, found none."


# ---------------------------------------------------------------------------
# train-ts command
# ---------------------------------------------------------------------------


def test_cli_train_time_series_smoke(
    time_series_config_path: Path,
) -> None:
    """train-ts (or train-time-series) should run and produce a model artifact."""
    paths = get_paths(config_path=time_series_config_path, force_reload=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    for f in paths.models_dir.iterdir():
        f.unlink()

    # We don't know if the command is named train-ts or train-time-series,
    # so we try the more compact one first and fall back if needed.
    for cmd_name in ("train-ts", "train-time-series"):
        result = runner.invoke(
            app,
            [
                cmd_name,
                "--config",
                str(time_series_config_path),
                "--device",
                "cpu",
                "--no-mlflow",
            ],
        )

        if result.exit_code == 0:
            break  # success
        last_result = result
    else:
        pytest.fail(
            f"Neither 'train-ts' nor 'train-time-series' CLI commands succeeded. "
            f"Last stdout/stderr:\n{last_result.stdout}"
        )

    # After success, we expect at least one model artifact
    model_files = list(paths.models_dir.glob("*.pt")) + list(
        paths.models_dir.glob("*.pth")
    )
    assert (
        len(model_files) >= 1
    ), f"Expected at least one .pt/.pth file in {paths.models_dir}, found none."


# ---------------------------------------------------------------------------
# Error behaviour
# ---------------------------------------------------------------------------


def test_cli_train_tabular_with_missing_config_fails(tmp_path: Path) -> None:
    """Passing a non-existent config path should result in a non-zero exit code."""
    missing_config = tmp_path / "does_not_exist.yaml"

    result = runner.invoke(
        app,
        [
            "train-tabular",
            "--config",
            str(missing_config),
            "--device",
            "cpu",
            "--no-mlflow",
        ],
    )

    # We don't force a particular error message format, only that it fails.
    assert result.exit_code != 0

    # It is nice (but not strictly required) if the CLI surfaces a helpful message.
    # We assert weakly here to avoid brittleness.
    stdout_lower = result.stdout.lower()
    assert (
        "not found" in stdout_lower
        or "no such file" in stdout_lower
        or "config" in stdout_lower
    )
