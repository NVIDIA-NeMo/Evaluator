# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the OpenClaw output-envelope extractor.

These lock down the shim that works around an upstream OpenClaw routing
bug where the ``--json`` result envelope lands on stderr instead of
stdout.  See ``_extract_openclaw_envelope`` for the full rationale.
"""

from __future__ import annotations

import json
import random
import time

import pytest

from nemo_evaluator.solvers.openclaw import (
    _MAX_SCAN_BYTES,
    _extract_openclaw_envelope,
    _iter_json_objects,
)


def _envelope(payloads: list[dict] | None = None, meta: dict | None = None) -> dict:
    return {
        "payloads": payloads if payloads is not None else [{"text": "hello"}],
        "meta": meta if meta is not None else {"durationMs": 42, "agentMeta": {"model": "m"}},
    }


def _serialized_envelope(**kwargs) -> str:
    """Match how OpenClaw's delivery.ts emits the envelope: pretty-printed, trailing newline."""
    return json.dumps(_envelope(**kwargs), indent=2) + "\n"


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestHappyPaths:
    def test_envelope_on_stdout_post_upstream_fix(self) -> None:
        """If upstream ever fixes delivery.ts, stdout is the documented channel."""
        stdout = _serialized_envelope()
        result = _extract_openclaw_envelope(stdout, "")
        assert result["payloads"] == [{"text": "hello"}]
        assert result["meta"]["durationMs"] == 42

    def test_envelope_on_stderr_only_with_subsystem_preamble(self) -> None:
        """The observed production pattern: stdout empty, envelope on stderr after log lines."""
        stderr = (
            "2026-04-15T10:00:00Z [agent/embedded] starting task\n"
            "2026-04-15T10:00:00Z [agent/embedded] model stream opened\n"
            "2026-04-15T10:00:05Z [agent/embedded] completed\n" + _serialized_envelope()
        )
        result = _extract_openclaw_envelope("", stderr)
        assert "payloads" in result
        assert result["meta"]["agentMeta"]["model"] == "m"

    def test_stdout_preferred_over_stderr(self) -> None:
        """When both streams carry an envelope, stdout wins."""
        stdout = _serialized_envelope(payloads=[{"text": "from-stdout"}])
        stderr = _serialized_envelope(payloads=[{"text": "from-stderr"}])
        result = _extract_openclaw_envelope(stdout, stderr)
        assert result["payloads"][0]["text"] == "from-stdout"

    def test_node_warning_before_stdout_envelope(self) -> None:
        """Node.js experimental-VM-modules warning must not block extraction."""
        stdout = (
            "(node:12345) ExperimentalWarning: VM Modules is an experimental feature\n"
            "(Use `node --trace-warnings ...` to show where the warning was created)\n" + _serialized_envelope()
        )
        result = _extract_openclaw_envelope(stdout, "")
        assert "payloads" in result


# ---------------------------------------------------------------------------
# Last-match discriminator
# ---------------------------------------------------------------------------


class TestLastMatchSemantics:
    def test_pre_envelope_log_with_payloads_does_not_hijack(self) -> None:
        """A structured log that itself has a ``payloads`` field must not win.

        We pick the LAST matching object, which — by construction — is the
        real envelope emitted right before command return.
        """
        fake_log = {"payloads": [{"text": "not-the-real-one"}], "source": "telemetry"}
        stderr = (
            "[agent/telemetry] preflight summary:\n"
            + json.dumps(fake_log)
            + "\n[agent/embedded] running\n"
            + _serialized_envelope(payloads=[{"text": "real-envelope"}])
        )
        result = _extract_openclaw_envelope("", stderr)
        assert result["payloads"][0]["text"] == "real-envelope"

    def test_log_line_mentioning_payloads_in_string_is_ignored(self) -> None:
        """A log line literally containing the word ``payloads`` in a string
        must not be parsed as the envelope."""
        stderr = "[agent/embedded] normalized 3 reply payloads in 12ms\n" + _serialized_envelope()
        result = _extract_openclaw_envelope("", stderr)
        assert result["payloads"] == [{"text": "hello"}]

    def test_non_payloads_json_objects_are_skipped(self) -> None:
        """JSON blobs without a ``payloads`` key must be skipped, not returned."""
        stderr = (
            json.dumps({"level": "info", "msg": "boot"})
            + "\n"
            + json.dumps({"probe": "ok", "duration": 3})
            + "\n"
            + _serialized_envelope()
        )
        result = _extract_openclaw_envelope("", stderr)
        assert result["meta"]["durationMs"] == 42


# ---------------------------------------------------------------------------
# Error / degenerate inputs
# ---------------------------------------------------------------------------


class TestDegenerateInputs:
    def test_empty_inputs_return_empty_dict(self) -> None:
        assert _extract_openclaw_envelope("", "") == {}

    def test_whitespace_only_returns_empty_dict(self) -> None:
        assert _extract_openclaw_envelope("\n\n", "   ") == {}

    def test_truncated_envelope_returns_empty(self) -> None:
        """A stream cut mid-object must yield ``{}``, not raise."""
        stderr = '{"payloads": [{"text": "hello"'
        assert _extract_openclaw_envelope("", stderr) == {}

    def test_non_object_json_ignored(self) -> None:
        """Bare arrays/scalars must not be treated as envelopes."""
        stderr = "[1, 2, 3]\n42\ntrue\n"
        assert _extract_openclaw_envelope("", stderr) == {}

    def test_unmatched_braces_do_not_confuse_scanner(self) -> None:
        stderr = "{ incomplete brace from truncated tool output\n" + _serialized_envelope()
        result = _extract_openclaw_envelope("", stderr)
        assert "payloads" in result


# ---------------------------------------------------------------------------
# Performance / safety
# ---------------------------------------------------------------------------


class TestPerformance:
    def test_realistic_large_stderr_completes_quickly(self) -> None:
        """A realistic ~500 KB stderr (many log lines, some embedded JSON)
        plus a real envelope must parse well under a second.

        Guards against regressions to the pre-``raw_decode`` quadratic scan:
        the old implementation scanned from every ``{`` to EOF on failure,
        which turned this exact shape into tens of seconds.
        """
        rng = random.Random(0)
        log_lines: list[str] = []
        approx_bytes = 0
        while approx_bytes < 500_000:
            if rng.random() < 0.2:
                line = "[agent/telemetry] " + json.dumps(
                    {"ts": rng.random(), "event": "probe", "ok": True},
                )
            else:
                line = "2026-04-20T12:00:00.000Z [agent/embedded] " + "tool output line " * rng.randint(2, 8)
            log_lines.append(line)
            approx_bytes += len(line) + 1
        stderr = "\n".join(log_lines) + "\n" + _serialized_envelope()
        start = time.perf_counter()
        result = _extract_openclaw_envelope("", stderr)
        elapsed = time.perf_counter() - start
        assert "payloads" in result
        assert elapsed < 1.0, f"parser took {elapsed:.3f}s (budget 1.0s)"

    def test_tail_cap_on_oversized_input(self) -> None:
        """Input beyond ``_MAX_SCAN_BYTES`` is tail-sliced; the envelope
        at the tail must still be found, and noise at the head must not
        make us time out."""
        padding = "x" * (_MAX_SCAN_BYTES + 100_000)
        stderr = padding + _serialized_envelope()
        start = time.perf_counter()
        result = _extract_openclaw_envelope("", stderr)
        elapsed = time.perf_counter() - start
        assert "payloads" in result
        assert elapsed < 0.3, f"parser took {elapsed:.3f}s (budget 0.3s)"

    def test_tail_cap_drops_head_envelope(self) -> None:
        """If the only envelope is beyond the tail cap from the end, we
        intentionally do not find it — better than OOM/timeout on
        runaway streams.  This pins that contract."""
        envelope = _serialized_envelope()
        stderr = envelope + "y" * (_MAX_SCAN_BYTES + 10_000)
        assert _extract_openclaw_envelope("", stderr) == {}


# ---------------------------------------------------------------------------
# _iter_json_objects
# ---------------------------------------------------------------------------


class TestIterJsonObjects:
    def test_skips_non_object_json(self) -> None:
        raw = '[1,2]\n{"a":1}\n"str"\n{"b":2}'
        objs = list(_iter_json_objects(raw))
        assert objs == [{"a": 1}, {"b": 2}]

    def test_handles_adjacent_objects(self) -> None:
        raw = '{"a":1}{"b":2}'
        objs = list(_iter_json_objects(raw))
        assert objs == [{"a": 1}, {"b": 2}]

    def test_string_literal_braces_are_not_mis_parsed(self) -> None:
        raw = '{"note": "contains { and } inside"}'
        objs = list(_iter_json_objects(raw))
        assert objs == [{"note": "contains { and } inside"}]


# ---------------------------------------------------------------------------
# Representative production fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def production_stderr() -> str:
    """Distilled fixture mirroring the pinchbench ECS failure signature."""
    return (
        "(node:7) ExperimentalWarning: VM Modules is an experimental feature\n"
        "2026-04-20T12:00:00.000Z [agent/embedded] openclaw starting\n"
        "2026-04-20T12:00:00.123Z [agent/model-auth] resolving provider auth\n"
        "2026-04-20T12:00:00.456Z [agent/embedded] model stream ready\n"
        "2026-04-20T12:00:05.789Z [agent/embedded] completed, 1 payload\n"
        + json.dumps(
            {
                "payloads": [{"text": "Hello, I'm ready!"}],
                "meta": {
                    "durationMs": 5333,
                    "usage": {"inputTokens": 12, "outputTokens": 8},
                    "agentMeta": {
                        "provider": "custom",
                        "model": "local",
                        "sessionId": "sess-1",
                    },
                },
            },
            indent=2,
        )
        + "\n"
    )


def test_matches_production_pattern(production_stderr: str) -> None:
    """End-to-end: the observed pinchbench failure shape must extract cleanly."""
    result = _extract_openclaw_envelope("", production_stderr)
    assert result["payloads"][0]["text"] == "Hello, I'm ready!"
    assert result["meta"]["durationMs"] == 5333
    assert result["meta"]["agentMeta"]["model"] == "local"
