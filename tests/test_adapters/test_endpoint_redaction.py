# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for endpoint.py's request-header redaction helper."""

from __future__ import annotations

from nemo_evaluator.adapters.interceptors.endpoint import _redact_request_headers_for_ctx


def test_drops_authorization_header_case_insensitively() -> None:
    out = _redact_request_headers_for_ctx(
        {
            "Authorization": "Bearer secret",
            "authorization": "Bearer also-secret",
            "X-Authorization-Token": "another-secret",
            "Content-Type": "application/json",
            "X-Inference-Priority": "batch",
        }
    )
    # All keys whose name contains "authorization" (any case) are gone.
    for k in out:
        assert "authorization" not in k.lower(), f"leaked: {k}"
    # Other headers pass through unchanged.
    assert out == {
        "Content-Type": "application/json",
        "X-Inference-Priority": "batch",
    }


def test_returns_shallow_copy_does_not_mutate_input() -> None:
    src = {"Authorization": "Bearer x", "Content-Type": "application/json"}
    out = _redact_request_headers_for_ctx(src)
    # Input untouched
    assert src == {"Authorization": "Bearer x", "Content-Type": "application/json"}
    # Output filtered
    assert out == {"Content-Type": "application/json"}


def test_no_op_on_already_clean_headers() -> None:
    src = {"Content-Type": "application/json", "X-Inference-Priority": "batch"}
    assert _redact_request_headers_for_ctx(src) == src
