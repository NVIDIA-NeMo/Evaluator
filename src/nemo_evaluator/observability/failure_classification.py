# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import re
from http import HTTPStatus
from typing import Any

MODEL_FAILURE_CATEGORIES = frozenset(
    {
        "turn_budget_exhausted",
        "context_window_exceeded",
        "budget_exceeded",
        "model_timeout",
        "rate_limit",
        "server_error",
        "model_error",
    }
)

_STATUS_RE = re.compile(
    r"(?:status[_ -]?code\s*[=:]\s*|status code\s*[=:]?\s*|error code:\s*|"
    r"http(?:/\d+(?:\.\d+)?)?\s+|status\s+)"
    r"(?P<code>[1-5][0-9]{2})\b",
    re.IGNORECASE,
)
_LEADING_STATUS_RE = re.compile(r"^\s*(?P<code>[1-5][0-9]{2})\s+(?P<reason>[^\r\n]+)")


def _status_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _status_from_text(text: str) -> int | None:
    if match := _STATUS_RE.search(text):
        return _status_int(match.group("code"))

    if not (match := _LEADING_STATUS_RE.search(text)):
        return None
    status = _status_int(match.group("code"))
    try:
        expected_reason = HTTPStatus(status).phrase
    except ValueError:
        return None
    reason = re.sub(r"[^a-z0-9]+", "", match.group("reason").lower())
    expected_reason = re.sub(r"[^a-z0-9]+", "", expected_reason.lower())
    return status if reason.startswith(expected_reason) else None


def _has_any(text: str, *markers: str) -> bool:
    return any(marker in text for marker in markers)


def classify_model_failure(error: str = "", *, status_code: Any = None) -> str | None:
    text = str(error or "").lower()
    status = _status_int(status_code)
    if status is None:
        status = _status_from_text(text)
    if not text and status is None:
        return None
    if status == 504:
        return "server_error"

    if _has_any(text, "turn budget exhausted", "turn budget exceeded"):
        return "turn_budget_exhausted"
    if _has_any(
        text,
        "contextwindowexceedederror",
        "context window exceeded",
        "context length exceeded",
        "context length is exceeded",
        "model's context length",
        "maximum context length",
        "maximum context",
        "too many tokens",
    ):
        return "context_window_exceeded"
    if _has_any(
        text,
        "budgetexceedederror",
        "token budget",
        "budget exceeded",
        "budget exhausted",
        "budget has been exceeded",
    ):
        return "budget_exceeded"
    if status == 429 or _has_any(
        text,
        "ratelimiterror",
        "routerratelimiterror",
        "routerratelimiterrorbasic",
        "ratelimittype",
        "rate_limit",
        "rate limit",
        "too many requests",
    ):
        return "rate_limit"
    if _has_any(text, "upstream timeout:", "gateway timeout:"):
        return "server_error"
    if status == 408 or _has_any(
        text,
        "apitimeouterror",
        "apiconnectiontimeouterror",
        "litellm.timeout",
        "timeout:",
        "timeouterror",
        "timeout error",
        "request timeout",
        "request timed out",
        "read timeout",
    ):
        return "model_timeout"
    if (status is not None and 500 <= status < 600) or _has_any(
        text,
        "badgatewayerror",
        "apiconnectionerror",
        "serviceunavailableerror",
        "internalservererror",
        "midstreamfallbackerror",
        "apiservererror",
        "api server error",
        "service unavailable",
        "bad gateway",
        "gateway timeout",
        "upstream timeout",
        "upstream timed out",
        "httpx.connecterror",
        "httpx.remoteprotocolerror",
        "serverdisconnectederror",
        "clientconnectorerror",
        "clientconnectionerror",
        "connectionreseterror",
        "server disconnected",
        "remote disconnected",
        "connection reset",
        "apierror",
        "no fallback model group found",
    ):
        return "server_error"
    if status == 400 or _has_any(
        text,
        "apiresponsevalidationerror",
        "jsonschemavalidationerror",
        "badrequesterror",
        "apistatuserror",
        "authenticationerror",
        "openaierror",
        "permissiondeniederror",
        "notfounderror",
        "unprocessableentityerror",
        "contentpolicyviolationerror",
        "unsupportedparamserror",
        "imagefetcherror",
        "litellmunknownprovider",
        "mockexception",
        "bad request",
        "invalid_request",
        "invalid request",
        "invalidrequesterror",
        "rejectedrequesterror",
        "blockedpiientityerror",
        "guardrailinterventionnormalstringerror",
        "guardrailraisedexception",
        "erroreventerror",
        "modifyresponseexception",
        "sensitivedatarouteexception",
        "malformed",
        "tool-call json",
        "tool call",
    ):
        return "model_error"
    if status is not None and not 200 <= status < 400:
        return "model_error"
    return None
