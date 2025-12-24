from __future__ import annotations

from typing import Any, Mapping


class AppError(Exception):
    """Base class for all application-specific exceptions.

    Attributes
    ----------
    message:
        Human-readable error message.
    code:
        Stable, machine-friendly identifier for this error type
        (e.g. "config_error", "data_error").
    cause:
        Optional underlying exception that triggered this error.
    context:
        Lightweight dictionary with extra debugging information.
    location:
        Optional string describing where the error occurred
        (e.g. "data_loader.load_csv", "train_tabular_mlp:main").
    """

    # Subclasses override this to give themselves a stable default code.
    default_code: str = "app_error"

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        cause: BaseException | None = None,
        context: Mapping[str, Any] | None = None,
        location: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        # If no explicit code is passed, use the class default.
        self.code: str = code or self.default_code
        self.cause = cause
        # Store a shallow copy so callers can pass any Mapping.
        self.context: dict[str, Any] = dict(context or {})
        self.location = location

        # Integrate with Python's exception chaining machinery.
        # This ensures traceback tools see the same underlying cause.
        if cause is not None:
            self.__cause__ = cause  # type: ignore[assignment]

    def __str__(self) -> str:
        parts: list[str] = [f"[{self.code}] {self.message}"]

        if self.location:
            parts.append(f"(at {self.location})")

        if self.cause is not None:
            parts.append(f"(cause: {self.cause!r})")

        return " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"code={self.code!r}, "
            f"message={self.message!r}, "
            f"location={self.location!r}"
            ")"
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the error."""
        data: dict[str, Any] = {
            "type": type(self).__name__,
            "code": self.code,
            "message": self.message,
        }

        if self.location:
            data["location"] = self.location

        if self.context:
            # Make a shallow copy to avoid accidental mutation from outside.
            data["context"] = dict(self.context)

        if self.cause is not None:
            data["cause"] = {
                "type": type(self.cause).__name__,
                "repr": repr(self.cause),
            }

        return data

    def add_context(self, **extra: Any) -> AppError:
        """Add or update context fields on this error and return self.

        This is handy when an error is being re-raised higher in the stack and
        you want to attach additional details (e.g. dataset name, run_id).
        """
        self.context.update(extra)
        return self

    @classmethod
    def from_exception(
        cls,
        exc: BaseException,
        *,
        message: str | None = None,
        code: str | None = None,
        context: Mapping[str, Any] | None = None,
        location: str | None = None,
    ) -> AppError:
        """Construct an AppError (or subclass) from an existing exception.

        Examples
        --------
        >>> try:
        ...     load_config(...)
        ... except Exception as exc:
        ...     raise ConfigError.from_exception(
        ...         exc,
        ...         message="Failed to parse YAML config",
        ...         code="config_parse_error",
        ...         context={"config_path": str(path)},
        ...         location="ml_tabular.config.load_config",
        ...     ) from exc
        """
        base_message = message or str(exc) or cls.__name__
        return cls(
            base_message,
            code=code,
            cause=exc,
            context=context,
            location=location,
        )


class ConfigError(AppError):
    """Configuration-related errors (missing keys, invalid values, etc.)."""

    default_code = "config_error"


class DataError(AppError):
    """Data loading, validation, or preprocessing errors."""

    default_code = "data_error"


class ModelError(AppError):
    """Errors during model training, evaluation, or inference."""

    default_code = "model_error"


class TrainingError(ModelError):
    """Errors raised during model training loops.

    Use this for:
    - invalid training arguments / hyperparameters,
    - failures inside training/eval loops,
    - unexpected NaNs/inf in loss, etc.
    """

    default_code = "training_error"


class PipelineError(AppError):
    """High-level orchestration / pipeline wiring errors."""

    default_code = "pipeline_error"
