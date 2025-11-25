from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from ml_tabular.mlops import mlflow_utils
from ml_tabular.mlops.mlflow_utils import (
    is_mlflow_available,
    mlflow_run,
    log_artifact,
    log_metrics,
    log_params,
)


# ---------------------------------------------------------------------------
# Availability detection
# ---------------------------------------------------------------------------


def test_is_mlflow_available_matches_runtime_import() -> None:
    """is_mlflow_available() should reflect whether mlflow can be imported."""
    try:
        import mlflow  # type: ignore[unused-import]
        expected = True
    except ImportError:
        expected = False

    assert is_mlflow_available() == expected


# ---------------------------------------------------------------------------
# No-op behaviour when disabled (works regardless of mlflow installation)
# ---------------------------------------------------------------------------


def test_mlflow_run_disabled_is_noop() -> None:
    """When enabled=False, mlflow_run should yield None and not raise."""
    with mlflow_run(
        enabled=False,
        experiment_name="dummy-experiment",
        run_name="dummy-run",
        tracking_uri=None,
        tags={"foo": "bar"},
    ) as run:
        # In disabled mode we don't expect a real run object
        assert run is None


def test_log_helpers_disabled_do_not_raise(tmp_path: Path) -> None:
    """log_params / log_metrics / log_artifact should quietly no-op when disabled."""
    # These calls should not raise, regardless of whether mlflow is installed.
    log_params({"a": 1, "b": "two"}, enabled=False)
    log_metrics({"loss": 0.123, "acc": 0.99}, step=5, enabled=False)

    dummy_file = tmp_path / "dummy.txt"
    dummy_file.write_text("hello", encoding="utf-8")

    log_artifact(dummy_file, artifact_path="artifacts", enabled=False)


# ---------------------------------------------------------------------------
# Behaviour when mlflow *is* installed
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not is_mlflow_available(),
    reason="mlflow is not installed; install mlflow and the 'mlops' extra to run these tests.",
)
class TestMlflowEnabled:
    """Tests that only run when mlflow is actually available."""

    def test_mlflow_run_creates_and_closes_run(self, tmp_path: Path) -> None:
        """mlflow_run(enabled=True, ...) should start and end an MLflow run cleanly."""
        import mlflow

        tracking_uri = tmp_path.as_uri()
        experiment_name = "test-experiment"

        with mlflow_run(
            enabled=True,
            experiment_name=experiment_name,
            run_name="test-run",
            tracking_uri=tracking_uri,
            tags={"purpose": "unit-test"},
        ) as run:
            # During the context, an active run should exist
            active = mlflow.active_run()
            assert active is not None
            assert run is not None
            assert active.info.run_id == run.info.run_id

            # We can log metrics/params within the context without error
            log_params({"alpha": 0.1, "beta": "test"}, enabled=True)
            log_metrics({"loss": 1.23}, step=1, enabled=True)

        # After leaving the context, there should be no active run
        assert mlflow.active_run() is None

    def test_log_params_delegates_to_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """log_params(enabled=True) should call mlflow.log_params with the same mapping."""
        import mlflow

        captured: Dict[str, Any] = {}

        def fake_log_params(params: Dict[str, Any]) -> None:
            captured["params"] = params

        monkeypatch.setattr(mlflow_utils.mlflow, "log_params", fake_log_params)

        params = {"lr": 0.001, "batch_size": 32}
        log_params(params, enabled=True)

        assert "params" in captured
        assert captured["params"] == params

    def test_log_metrics_delegates_to_mlflow(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """log_metrics(enabled=True) should call mlflow.log_metrics (or log_metric in a loop)."""
        import mlflow

        # We don't know if your wrapper uses log_metrics or log_metric in a loop,
        # so we patch both and assert at least one is used.
        captured_metrics: Dict[str, Any] = {"log_metric_calls": [], "log_metrics_calls": []}

        def fake_log_metric(key: str, value: float, step: int | None = None) -> None:
            captured_metrics["log_metric_calls"].append((key, value, step))

        def fake_log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
            captured_metrics["log_metrics_calls"].append((metrics, step))

        # Patch both; your implementation should use one of them
        monkeypatch.setattr(mlflow_utils.mlflow, "log_metric", fake_log_metric, raising=False)
        monkeypatch.setattr(mlflow_utils.mlflow, "log_metrics", fake_log_metrics, raising=False)

        metrics = {"loss": 0.5, "accuracy": 0.9}
        log_metrics(metrics, step=3, enabled=True)

        # At least one of the patched functions should have been called
        used_log_metric = len(captured_metrics["log_metric_calls"]) > 0
        used_log_metrics = len(captured_metrics["log_metrics_calls"]) > 0
        assert used_log_metric or used_log_metrics

        if used_log_metrics:
            logged_metrics, step = captured_metrics["log_metrics_calls"][-1]
            assert logged_metrics == metrics
            assert step == 3
        elif used_log_metric:
            # For log_metric per-key, check that keys/values look right
            calls = captured_metrics["log_metric_calls"]
            keys = {k for k, _, _ in calls}
            assert keys == set(metrics.keys())

    def test_log_artifact_delegates_to_mlflow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """log_artifact(enabled=True) should forward to mlflow.log_artifact."""
        import mlflow

        captured: Dict[str, Any] = {}

        def fake_log_artifact(local_path: str, artifact_path: str | None = None) -> None:
            captured["local_path"] = local_path
            captured["artifact_path"] = artifact_path

        monkeypatch.setattr(mlflow_utils.mlflow, "log_artifact", fake_log_artifact)

        # Create a dummy file to log
        dummy = tmp_path / "model.pt"
        dummy.write_bytes(b"fake model")

        log_artifact(dummy, artifact_path="models", enabled=True)

        assert captured.get("local_path") == str(dummy)
        assert captured.get("artifact_path") == "models"

    def test_log_helpers_respect_enabled_flag_even_when_mlflow_present(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Even when mlflow is installed, enabled=False should prevent calls."""
        import mlflow

        called = {
            "log_params": False,
            "log_metrics": False,
            "log_artifact": False,
        }

        def fake_log_params(params: Dict[str, Any]) -> None:
            called["log_params"] = True

        def fake_log_metrics(metrics: Dict[str, float], step: int | None = None) -> None:
            called["log_metrics"] = True

        def fake_log_artifact(local_path: str, artifact_path: str | None = None) -> None:
            called["log_artifact"] = True

        monkeypatch.setattr(mlflow_utils.mlflow, "log_params", fake_log_params, raising=False)
        monkeypatch.setattr(mlflow_utils.mlflow, "log_metrics", fake_log_metrics, raising=False)
        monkeypatch.setattr(mlflow_utils.mlflow, "log_artifact", fake_log_artifact, raising=False)

        dummy = tmp_path / "dummy.txt"
        dummy.write_text("hello", encoding="utf-8")

        log_params({"x": 1}, enabled=False)
        log_metrics({"loss": 1.0}, step=1, enabled=False)
        log_artifact(dummy, artifact_path=None, enabled=False)

        assert called == {
            "log_params": False,
            "log_metrics": False,
            "log_artifact": False,
        }
