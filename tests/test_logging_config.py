from __future__ import annotations

import logging

import pytest

from ml_tabular.logging_config import get_logger


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------


def test_get_logger_returns_configured_logger() -> None:
    """get_logger should return a Logger with at least one handler and a formatter."""
    name = "ml_tabular.tests.logging_basic"
    logger = get_logger(name)

    # Correct type and name
    assert isinstance(logger, logging.Logger)
    assert logger.name == name

    # At least one handler is attached
    assert logger.handlers, "Expected at least one handler to be attached to the logger."

    # Each handler should have a formatter (we don't assert exact format string to avoid brittleness)
    for handler in logger.handlers:
        assert handler.formatter is not None, "Expected handler to have a formatter configured."
        fmt = handler.formatter._style._fmt  # type: ignore[attr-defined]
        # Gentle sanity checks: typical fields you set in your logging_config
        assert "levelname" in fmt or "%(levelname" in fmt
        assert "name" in fmt or "%(name" in fmt
        # asctime is optional but common; don't require it strictly


def test_get_logger_is_idempotent_for_same_name() -> None:
    """Calling get_logger with the same name should not create duplicate handlers."""
    name = "ml_tabular.tests.logging_idempotent"

    logger1 = get_logger(name)
    handlers_before = list(logger1.handlers)

    logger2 = get_logger(name)
    handlers_after = list(logger2.handlers)

    # Same logger instance
    assert logger1 is logger2

    # Same handlers, no duplicates added
    assert handlers_after == handlers_before
    assert len(handlers_after) == len(set(handlers_after))


# ---------------------------------------------------------------------------
# Logging output behaviour
# ---------------------------------------------------------------------------


def test_logger_emits_info_messages(caplog: pytest.LogCaptureFixture) -> None:
    """get_logger should produce INFO-level logs that are visible via caplog."""
    name = "ml_tabular.tests.logging_emit"
    logger = get_logger(name)

    message = "hello from logging test"

    with caplog.at_level(logging.INFO, logger=name):
        logger.info(message)

    # There should be at least one record with our message
    assert any(
        record.getMessage() == message and record.levelno == logging.INFO
        for record in caplog.records
    ), "Expected INFO log message to appear in captured logs."


def test_logger_respects_log_level_for_debug(caplog: pytest.LogCaptureFixture) -> None:
    """If we request DEBUG level in caplog, debug messages should appear."""
    name = "ml_tabular.tests.logging_debug"
    logger = get_logger(name)

    debug_message = "debug message from logging test"

    with caplog.at_level(logging.DEBUG, logger=name):
        logger.debug(debug_message)

    assert any(
        record.getMessage() == debug_message and record.levelno == logging.DEBUG
        for record in caplog.records
    ), "Expected DEBUG log message to appear when level is set to DEBUG."


# ---------------------------------------------------------------------------
# Handler properties (non-brittle checks)
# ---------------------------------------------------------------------------


def test_logger_handlers_are_stream_handlers_by_default() -> None:
    """For a library-style logger, stream handlers are a sensible default.

    This test doesn't enforce an exact handler class, but asserts that
    at least one handler is any subclass of logging.Handler and that no
    obviously odd handler types are present.
    """
    name = "ml_tabular.tests.logging_handlers"
    logger = get_logger(name)

    assert logger.handlers, "Expected at least one handler."

    # All handlers should be logging.Handler subclasses
    for handler in logger.handlers:
        assert isinstance(handler, logging.Handler)

    # At least one handler should be a StreamHandler (typical for console logging)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
