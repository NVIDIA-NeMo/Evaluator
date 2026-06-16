# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the chat-template-on-text-completions guard."""

from __future__ import annotations

import logging

import pytest

from nemo_evaluator.completions_guard import (
    ALLOW_ENV_VAR,
    allow_completions_chat_template,
    enforce_text_completions_body,
    is_text_completions_endpoint_type,
    is_text_completions_path,
    normalize_adapter_config_for_endpoint,
)
from nemo_evaluator.environments.container import build_legacy_run_config


@pytest.fixture(autouse=True)
def _clear_allow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)


# ── predicates ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/v1/completions", True),
        ("/v1/completions/", True),
        ("/completions", True),
        ("/v1/chat/completions", False),
        ("/v1/chat/completions/", False),
        ("/v1/embeddings", False),
        ("/v1/responses", False),
    ],
)
def test_is_text_completions_path(path: str, expected: bool) -> None:
    assert is_text_completions_path(path) is expected


@pytest.mark.parametrize(
    ("endpoint_type", "expected"),
    [
        ("completions", True),
        ("Completions", True),
        (" completions ", True),
        ("chat", False),
        ("chat_completions", False),
        (None, False),
        ("", False),
    ],
)
def test_is_text_completions_endpoint_type(endpoint_type: str | None, expected: bool) -> None:
    assert is_text_completions_endpoint_type(endpoint_type) is expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("false", False), ("", False)],
)
def test_allow_completions_chat_template(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(ALLOW_ENV_VAR, value)
    assert allow_completions_chat_template() is expected


# ── request body (#2, in-container send path) ────────────────────────────────


@pytest.mark.parametrize(
    "toggle",
    [
        {"enable_thinking": False},
        {"thinking": False},
        {"thinking": True},
        {"enable_thinking": True},
        {"nested": {"x": False}},
        {"nested": {"x": True}},
    ],
)
def test_body_hard_fails_on_boolean_toggle(toggle: dict) -> None:
    body = {"prompt": "rendered", "chat_template_kwargs": toggle}
    with pytest.raises(ValueError) as exc_info:
        enforce_text_completions_body("/v1/completions", body)
    # The error must tell the user how to deactivate the failure.
    assert "boolean toggle" in str(exc_info.value)
    assert ALLOW_ENV_VAR in str(exc_info.value)
    # It raises before mutating the body.
    assert "chat_template_kwargs" in body


def test_body_hard_fails_on_nested_boolean_toggle() -> None:
    body = {"prompt": "rendered", "extra_body": {"chat_template_kwargs": {"thinking": True}}}
    with pytest.raises(ValueError, match="boolean toggle"):
        enforce_text_completions_body("/v1/completions", body)


@pytest.mark.parametrize("toggle", [{"thinking": True}, {"enable_thinking": False}])
def test_body_allow_env_downgrades_boolean_to_warning(
    toggle: dict, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(ALLOW_ENV_VAR, "1")
    body = {"prompt": "rendered", "chat_template_kwargs": toggle}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}
    assert "chat_template_kwargs" in caplog.text
    assert "params.extra.args" in caplog.text


@pytest.mark.parametrize("kwargs", [{"thinking_mode": "chat"}, {"reasoning_effort": "max"}])
def test_body_strips_and_warns_on_non_boolean(kwargs: dict, caplog: pytest.LogCaptureFixture) -> None:
    body = {"prompt": "rendered", "chat_template_kwargs": kwargs}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}
    assert "chat_template_kwargs" in caplog.text
    assert "params.extra.args" in caplog.text


def test_body_integer_is_not_a_boolean(caplog: pytest.LogCaptureFixture) -> None:
    # 0/1 are ints, not booleans: no toggle, so strip+warn rather than hard fail.
    body = {"prompt": "rendered", "chat_template_kwargs": {"some_count": 0, "other": 1}}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}


def test_body_chat_template_string_strips_not_fails(caplog: pytest.LogCaptureFixture) -> None:
    body = {"prompt": "rendered", "chat_template": "{{ jinja }}"}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            {"prompt": "rendered", "extra_body": {"chat_template_kwargs": {"thinking": True}}},
            {"prompt": "rendered", "extra_body": {}},
        ),
        (
            {"prompt": "rendered", "items": [{"chat_template_kwargs": {"thinking": True}}]},
            {"prompt": "rendered", "items": [{}]},
        ),
    ],
    ids=["nested-dict", "nested-list"],
)
def test_body_strips_nested_fields_when_allowed(body: dict, expected: dict, caplog: pytest.LogCaptureFixture) -> None:
    # Recursive stripping: with the allow opt-out, nested boolean toggles are
    # removed everywhere they were detected (not just at the top level).
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body, allow=True)
    assert body == expected
    assert "chat_template" in caplog.text
    assert "params.extra.args" in caplog.text


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/chat/completions/"])
def test_body_allows_chat_endpoint(path: str) -> None:
    body = {"messages": [], "chat_template_kwargs": {"enable_thinking": False}}
    enforce_text_completions_body(path, body)
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_body_noop_on_clean_request() -> None:
    body = {"prompt": "rendered", "max_tokens": 16}
    enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered", "max_tokens": 16}


# ── adapter_config (#1, launcher/config-assembly path) ────────────────────────


@pytest.mark.parametrize("toggle", [{"enable_thinking": False}, {"thinking": True}])
def test_adapter_config_hard_fails_on_boolean_toggle(toggle: dict) -> None:
    adapter_config = {"params_to_add": {"chat_template_kwargs": toggle}}
    with pytest.raises(ValueError) as exc_info:
        normalize_adapter_config_for_endpoint(adapter_config, "completions")
    assert "boolean toggle" in str(exc_info.value)
    assert ALLOW_ENV_VAR in str(exc_info.value)


def test_adapter_config_allow_env_strips_inherited_toggle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    # Mirrors the real deepseek RULER merge: _model.yaml injects thinking: true.
    # With the allow opt-out the inherited toggle is stripped instead of failing.
    monkeypatch.setenv(ALLOW_ENV_VAR, "1")
    adapter_config = {
        "use_reasoning": False,
        "process_reasoning_traces": False,
        "params_to_add": {
            "chat_template_kwargs": {"thinking": True, "reasoning_effort": "max"},
            "skip_special_tokens": False,
            "min_tokens": 4,
        },
    }
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        cleaned = normalize_adapter_config_for_endpoint(adapter_config, "completions")
    assert cleaned == {
        "use_reasoning": False,
        "process_reasoning_traces": False,
        "params_to_add": {"skip_special_tokens": False, "min_tokens": 4},
    }
    assert adapter_config["params_to_add"]["chat_template_kwargs"] == {"thinking": True, "reasoning_effort": "max"}
    assert "params_to_add.chat_template_kwargs" in caplog.text


def test_adapter_config_strips_non_boolean_by_default(caplog: pytest.LogCaptureFixture) -> None:
    # A non-boolean chat-template value carries no on/off intent: strip+warn, no error.
    adapter_config = {"params_to_add": {"chat_template_kwargs": {"thinking_mode": "chat"}}}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        cleaned = normalize_adapter_config_for_endpoint(adapter_config, "completions")
    assert cleaned == {"params_to_add": {}}
    assert "params_to_add.chat_template_kwargs" in caplog.text


def test_adapter_config_identity_on_chat_endpoint() -> None:
    adapter_config = {"params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}}
    result = normalize_adapter_config_for_endpoint(adapter_config, "chat")
    assert result is adapter_config


def test_adapter_config_identity_when_clean() -> None:
    adapter_config = {"params_to_add": {"temperature": 0.0}}
    result = normalize_adapter_config_for_endpoint(adapter_config, "completions")
    assert result is adapter_config


def test_adapter_config_none_passthrough() -> None:
    assert normalize_adapter_config_for_endpoint(None, "completions") is None


# ── build_legacy_run_config integration ──────────────────────────────────────


def test_build_legacy_run_config_hard_fails_on_boolean(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(ALLOW_ENV_VAR, raising=False)
    with pytest.raises(ValueError, match="boolean toggle"):
        build_legacy_run_config(
            task="ruler",
            model_url="http://localhost:8000/v1/completions",
            model_id="m",
            endpoint_type="completions",
            adapter_config={"params_to_add": {"chat_template_kwargs": {"thinking": True}}},
        )


def test_build_legacy_run_config_allow_env_strips_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ALLOW_ENV_VAR, "1")
    run_config = build_legacy_run_config(
        task="ruler",
        model_url="http://localhost:8000/v1/completions",
        model_id="m",
        endpoint_type="completions",
        adapter_config={"params_to_add": {"chat_template_kwargs": {"thinking": True}, "seed": 1}},
    )
    embedded = run_config["target"]["api_endpoint"]["adapter_config"]
    assert embedded == {"params_to_add": {"seed": 1}}


def test_build_legacy_run_config_keeps_for_chat() -> None:
    adapter_config = {"params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}}
    run_config = build_legacy_run_config(
        task="mmlu",
        model_url="http://localhost:8000/v1/chat/completions",
        model_id="m",
        endpoint_type="chat",
        adapter_config=adapter_config,
    )
    embedded = run_config["target"]["api_endpoint"]["adapter_config"]
    assert embedded == adapter_config
