"""Exception helpers shared by evaluator resilience/task flows."""

from jinja2.exceptions import UndefinedError as JinjaUndefinedError


def first_failure_cause(exc: BaseException) -> BaseException:
    """Return the first leaf failure from an exception/exception-group tree."""
    current = exc
    while hasattr(current, "exceptions"):
        exceptions = getattr(current, "exceptions", ())
        if not isinstance(exceptions, tuple) or not exceptions:
            break
        candidate = exceptions[0]
        if not isinstance(candidate, BaseException):
            break
        current = candidate
    return current


def normalize_evaluation_failure(
    exc: BaseException,
    *,
    prefix: str = "Metric evaluation has failed",
) -> RuntimeError:
    """Convert queue/task execution failures into the public evaluator error shape."""
    root = first_failure_cause(exc)
    if isinstance(root, JinjaUndefinedError):
        return RuntimeError(f"{prefix} due to templating error: {str(root)}")
    if isinstance(exc, JinjaUndefinedError):
        return RuntimeError(f"{prefix} due to templating error: {str(exc)}")
    return RuntimeError(f"{prefix} with error: {str(root) or root.__class__.__name__}")
