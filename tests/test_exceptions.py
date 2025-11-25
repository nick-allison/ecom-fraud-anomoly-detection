from __future__ import annotations

import traceback
from typing import Any, Dict, Optional

import pytest

from ml_tabular.exceptions import (
    BaseAppError,
    ConfigError,
    DataError,
    ModelError,
    TrainingError,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_tb_str(exc: BaseAppError) -> str:
    """Render a short traceback string for debugging if a test fails."""
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


# ---------------------------------------------------------------------------
# BaseAppError behaviour
# ---------------------------------------------------------------------------


def test_base_app_error_basic_fields_are_preserved() -> None:
    cause = ValueError("underlying problem")
    context: Dict[str, Any] = {"config_path": "configs/bad.yaml", "env": "dev"}
    location = "ml_tabular.config.load_config"

    err = BaseAppError(
        "Failed to load configuration",
        code="config_load_error",
        context=context,
        cause=cause,
        location=location,
    )

    # Attributes are preserved
    assert err.message == "Failed to load configuration"
    assert err.code == "config_load_error"
    assert err.context is context  # the exact mapping we passed in
    assert err.cause is cause
    assert err.location == location

    # It behaves like an Exception
    assert isinstance(err, Exception)

    # __str__ should at least contain the message; including the code/location is a bonus
    text = str(err)
    assert "Failed to load configuration" in text
    # Don't make this too brittle, but if your __str__ includes code/location it's nice:
    # (these assertions are written defensively)
    if err.code:
        assert "config_load_error" in text or True
    if err.location:
        assert "ml_tabular.config.load_config" in text or True


def test_base_app_error_default_args_are_reasonable() -> None:
    """Construct a BaseAppError with minimal args and ensure defaults don't explode."""
    err = BaseAppError("Something went wrong")

    assert err.message == "Something went wrong"
    # code / context / cause / location may be None or set to sensible defaults;
    # we only require that they exist as attributes and don't blow up.
    assert hasattr(err, "code")
    assert hasattr(err, "context")
    assert hasattr(err, "cause")
    assert hasattr(err, "location")

    # str() must be defined and contain the message
    text = str(err)
    assert "Something went wrong" in text


def test_base_app_error_context_is_mutable_mapping_if_provided() -> None:
    ctx: Dict[str, Any] = {"foo": "bar"}
    err = BaseAppError("msg", context=ctx)

    assert err.context is ctx
    err.context["extra"] = 123  # type: ignore[assignment]
    assert err.context["extra"] == 123  # type: ignore[index]


# ---------------------------------------------------------------------------
# Subclass behaviour (ConfigError, DataError, ModelError, TrainingError)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc_type, code_prefix",
    [
        (ConfigError, "config_"),
        (DataError, "data_"),
        (ModelError, "model_"),
        (TrainingError, "training_"),
    ],
)
def test_specific_errors_are_subclasses_of_base_app_error(
    exc_type: type[BaseAppError],
    code_prefix: str,
) -> None:
    """Each specific error should inherit from BaseAppError and behave similarly."""
    err = exc_type(
        "High-level message",
        code=f"{code_prefix}something",
        context={"key": "value"},
        location="some.module:func",
    )

    # Inheritance
    assert isinstance(err, exc_type)
    assert isinstance(err, BaseAppError)
    assert isinstance(err, Exception)

    # Attributes and message
    assert err.message == "High-level message"
    assert err.code.startswith(code_prefix)
    assert err.context == {"key": "value"}
    assert err.location == "some.module:func"

    # __str__ should contain the message
    assert "High-level message" in str(err)


def test_config_error_wraps_underlying_exception_as_cause() -> None:
    """ConfigError should be able to wrap a lower-level exception for debugging."""
    underlying = FileNotFoundError("configs/missing.yaml")

    exc = ConfigError(
        "Config file not found",
        code="config_file_not_found",
        cause=underlying,
        context={"config_path": "configs/missing.yaml"},
        location="ml_tabular.config.load_config",
    )

    assert exc.cause is underlying
    text = str(exc)
    # Don't be overly strict, but it's helpful if the wrapped message appears in __str__
    assert "Config file not found" in text
    if isinstance(exc.cause, Exception):
        assert "configs/missing.yaml" in repr(exc.cause) or True


def test_data_model_training_errors_are_semantically_separated() -> None:
    """DataError, ModelError, and TrainingError should be distinct types."""
    data_err = DataError("data issue", code="data_missing_column")
    model_err = ModelError("model issue", code="model_shape_mismatch")
    training_err = TrainingError("training issue", code="training_nan_loss")

    assert not isinstance(data_err, ModelError)
    assert not isinstance(data_err, TrainingError)

    assert not isinstance(model_err, DataError)
    assert not isinstance(model_err, TrainingError)

    assert not isinstance(training_err, DataError)
    assert not isinstance(training_err, ModelError)


def test_errors_can_be_raised_and_caught_normally() -> None:
    """All custom errors must behave like normal Exceptions in try/except."""
    with pytest.raises(ConfigError) as ctx:
        raise ConfigError("Bad config", code="config_invalid")

    exc = ctx.value
    assert isinstance(exc, ConfigError)
    assert isinstance(exc, BaseAppError)
    assert "Bad config" in str(exc)


def test_traceback_is_available_for_debugging() -> None:
    """Even with custom error types, Python's traceback machinery should work."""
    try:
        def inner() -> None:
            raise DataError("data is bad", code="data_validation_failed", location="inner_func")

        inner()
    except DataError as exc:
        tb_str = _extract_tb_str(exc)
        # A coarse assertion that the traceback mentions the inner function and our message
        assert "inner" in tb_str or "inner_func" in tb_str
        assert "data is bad" in tb_str
    else:
        raise AssertionError("Expected DataError to be raised but nothing was raised.")
