# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the chat-template-on-text-completions guard."""

from __future__ import annotations

import logging

import pytest

from nemo_evaluator.completions_guard import (
    STRICT_ENV_VAR,
    enforce_text_completions_body,
    is_text_completions_endpoint_type,
    is_text_completions_path,
    normalize_adapter_config_for_endpoint,
    strict_completions_enabled,
)
from nemo_evaluator.environments.container import build_legacy_run_config


@pytest.fixture(autouse=True)
def _clear_strict_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(STRICT_ENV_VAR, raising=False)


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
def test_strict_completions_enabled(monkeypatch: pytest.MonkeyPatch, value: str, expected: bool) -> None:
    monkeypatch.setenv(STRICT_ENV_VAR, value)
    assert strict_completions_enabled() is expected


# ── request body (#2, in-container send path) ────────────────────────────────


@pytest.mark.parametrize("disable", [{"enable_thinking": False}, {"thinking": False}, {"nested": {"x": False}}])
def test_body_hard_fails_on_disable_intent(disable: dict) -> None:
    body = {"prompt": "rendered", "chat_template_kwargs": disable}
    with pytest.raises(ValueError, match="disabling value"):
        enforce_text_completions_body("/v1/completions", body)
    assert "chat_template_kwargs" in body


@pytest.mark.parametrize(
    "enabling",
    [{"thinking": True, "reasoning_effort": "max"}, {"thinking_mode": "chat"}, {"enable_thinking": True}],
)
def test_body_strips_and_warns_on_non_disabling(enabling: dict, caplog: pytest.LogCaptureFixture) -> None:
    body = {"prompt": "rendered", "chat_template_kwargs": enabling}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}
    assert "chat_template_kwargs" in caplog.text
    assert "params.extra.args" in caplog.text


def test_body_integer_zero_is_not_a_disable(caplog: pytest.LogCaptureFixture) -> None:
    body = {"prompt": "rendered", "chat_template_kwargs": {"some_count": 0}}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}


def test_body_strict_hard_fails_even_when_enabling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    body = {"prompt": "rendered", "chat_template_kwargs": {"thinking": True}}
    with pytest.raises(ValueError, match="STRICT"):
        enforce_text_completions_body("/v1/completions", body)


def test_body_chat_template_string_strips_not_fails(caplog: pytest.LogCaptureFixture) -> None:
    body = {"prompt": "rendered", "chat_template": "{{ jinja }}"}
    with caplog.at_level(logging.WARNING, logger="nemo_evaluator.completions_guard"):
        enforce_text_completions_body("/v1/completions", body)
    assert body == {"prompt": "rendered"}


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


def test_adapter_config_hard_fails_on_nested_disable() -> None:
    adapter_config = {"params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}}
    with pytest.raises(ValueError, match="disabling value"):
        normalize_adapter_config_for_endpoint(adapter_config, "completions")


def test_adapter_config_strips_inherited_enabling_value(caplog: pytest.LogCaptureFixture) -> None:
    # Mirrors the real deepseek RULER merge: _model.yaml injects thinking: true.
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


def test_adapter_config_strict_hard_fails_even_when_enabling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(STRICT_ENV_VAR, "1")
    adapter_config = {"params_to_add": {"chat_template_kwargs": {"thinking": True}}}
    with pytest.raises(ValueError, match="STRICT"):
        normalize_adapter_config_for_endpoint(adapter_config, "completions")


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


def test_build_legacy_run_config_strips_inherited_enabling() -> None:
    run_config = build_legacy_run_config(
        task="ruler",
        model_url="http://localhost:8000/v1/completions",
        model_id="m",
        endpoint_type="completions",
        adapter_config={"params_to_add": {"chat_template_kwargs": {"thinking": True}, "seed": 1}},
    )
    embedded = run_config["target"]["api_endpoint"]["adapter_config"]
    assert embedded == {"params_to_add": {"seed": 1}}


def test_build_legacy_run_config_hard_fails_on_disable() -> None:
    with pytest.raises(ValueError, match="disabling value"):
        build_legacy_run_config(
            task="ruler",
            model_url="http://localhost:8000/v1/completions",
            model_id="m",
            endpoint_type="completions",
            adapter_config={"params_to_add": {"chat_template_kwargs": {"enable_thinking": False}}},
        )


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
