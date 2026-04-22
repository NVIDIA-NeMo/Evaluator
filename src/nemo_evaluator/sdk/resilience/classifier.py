"""Exception classification and endpoint identity helpers for resilience."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime

import httpx
import openai
from nemo_evaluator.sdk.resilience.types import ClassifierResult, FailureClass

_HARD_OVERLOAD_STATUS_CODES = frozenset({429, 503})
_TRANSIENT_STATUS_CODES = frozenset({408, 500, 502, 504})


def endpoint_identity(base_url: str, model_id: str | None = None, auth_identity: str | None = None) -> str:
    """Build a stable endpoint key for scheduler state and accounting."""
    auth_hash = ""
    if auth_identity:
        auth_hash = hashlib.sha256(auth_identity.encode("utf-8")).hexdigest()[:16]
    return f"{base_url}|{model_id or '_'}|{auth_hash}"


def _status_code_from_exception(exc: Exception) -> int | None:
    """Return HTTP status code if present on a known exception type."""
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code
    return None


def _parse_retry_after_value(value: str | None) -> float | None:
    """Parse `Retry-After` header values into seconds."""
    if value is None:
        return None
    raw = value.strip()
    try:
        parsed = float(raw)
    except ValueError:
        try:
            when = parsedate_to_datetime(raw)
        except (TypeError, ValueError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        parsed = (when - datetime.now(tz=UTC)).total_seconds()
    if parsed < 0:
        return 0.0
    return parsed


def _retry_after_from_headers(headers: httpx.Headers | None) -> float | None:
    """Extract and parse `Retry-After` from an HTTP header mapping."""
    if not headers:
        return None
    return _parse_retry_after_value(headers.get("Retry-After") or headers.get("retry-after"))


def _retry_after_from_exception(exc: Exception) -> float | None:
    """Extract `Retry-After` seconds from supported exception types."""
    if isinstance(exc, httpx.HTTPStatusError):
        return _retry_after_from_headers(exc.response.headers)
    if isinstance(exc, openai.APIStatusError):
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if isinstance(headers, httpx.Headers):
            return _retry_after_from_headers(headers)
        if isinstance(headers, dict):
            return _parse_retry_after_value(headers.get("Retry-After") or headers.get("retry-after"))
    return None


def classify_exception(exc: Exception) -> ClassifierResult:
    """Classify errors into retry/failure policy buckets."""
    retry_after = _retry_after_from_exception(exc)
    status_code = _status_code_from_exception(exc)
    error_type = type(exc).__name__

    if isinstance(exc, (httpx.ReadTimeout, httpx.WriteTimeout, httpx.PoolTimeout, openai.APITimeoutError)):
        return ClassifierResult(
            failure_class=FailureClass.SOFT_OVERLOAD,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )
    if isinstance(exc, httpx.ConnectTimeout):
        return ClassifierResult(
            failure_class=FailureClass.TRANSIENT,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )
    if isinstance(exc, (httpx.NetworkError, openai.APIConnectionError)):
        return ClassifierResult(
            failure_class=FailureClass.TRANSIENT,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )

    if status_code in _HARD_OVERLOAD_STATUS_CODES:
        return ClassifierResult(
            failure_class=FailureClass.HARD_OVERLOAD,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )
    if status_code in _TRANSIENT_STATUS_CODES:
        return ClassifierResult(
            failure_class=FailureClass.TRANSIENT,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )

    # Rate-limit style errors without explicit status code handling above.
    if isinstance(exc, (openai.RateLimitError,)):
        return ClassifierResult(
            failure_class=FailureClass.HARD_OVERLOAD,
            retryable=True,
            retry_after_seconds=retry_after,
            status_code=status_code,
            error_type=error_type,
        )

    return ClassifierResult(
        failure_class=FailureClass.FATAL,
        retryable=False,
        retry_after_seconds=retry_after,
        status_code=status_code,
        error_type=error_type,
    )
