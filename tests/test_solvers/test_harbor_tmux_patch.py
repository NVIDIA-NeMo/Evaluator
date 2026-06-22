# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Tests for the Harbor terminus-2 ``tmux send-keys`` UTF-8 byte-length monkeypatch.

These verify ``_patch_terminus_tmux_send_keys`` makes Harbor chunk keystrokes by
UTF-8 byte length (tmux's real limit) instead of Unicode code-point count, which
otherwise lets multibyte payloads (e.g. Lean proofs) exceed the limit and crash
the agent with ``command too long``.

Harbor is an optional dependency, so the suite skips when it isn't installed.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

from nemo_evaluator.solvers.harbor import _patch_terminus_tmux_send_keys

# Skip the whole module unless Harbor's terminus-2 TmuxSession is importable.
_tmux_mod = pytest.importorskip("harbor.agents.terminus_2.tmux_session")
TmuxSession = _tmux_mod.TmuxSession

LEAN4_PROOF_STEP33_FIXTURE = Path(__file__).parent / "fixtures" / "lean4_proof_repeat0_step33_keystrokes.txt"

SESSION_NAME = "test-session"


@pytest.fixture(autouse=True)
def _apply_patch():
    """Ensure the byte-length patch is applied before each test."""
    _patch_terminus_tmux_send_keys()


def _make_session(session_name: str = SESSION_NAME) -> TmuxSession:
    """Build a TmuxSession without running Harbor's heavy ``__init__``.

    ``_tmux_send_keys`` only reads ``_session_name`` and the class-level
    ``_TMUX_SEND_KEYS_MAX_COMMAND_LENGTH``, so a bare instance suffices.
    """
    session = TmuxSession.__new__(TmuxSession)
    session._session_name = session_name
    return session


def _command_len(command: str) -> int:
    return len(command.encode("utf-8"))


def _assert_commands_below_limit(session: TmuxSession, commands: list[str]) -> None:
    assert all(_command_len(command) <= session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH for command in commands)


def _extract_send_keys_payload(command: str, session_name: str = SESSION_NAME) -> list[str]:
    parts = shlex.split(command)
    assert parts[:4] == ["tmux", "send-keys", "-t", session_name]
    return parts[4:]


def _extract_all_send_keys_payload(commands: list[str], session_name: str = SESSION_NAME) -> list[str]:
    payload: list[str] = []
    for command in commands:
        payload.extend(_extract_send_keys_payload(command, session_name=session_name))
    return payload


# ── Patch application ────────────────────────────────────────────────────


class TestPatchApplication:
    def test_patch_installs_byte_length_methods(self):
        assert getattr(TmuxSession, "_nel_tmux_bytelen_patched", False) is True
        assert TmuxSession._utf8_len("é") == 2
        assert TmuxSession._utf8_len("abc") == 3

    def test_patch_is_idempotent(self):
        first = TmuxSession._tmux_send_keys
        _patch_terminus_tmux_send_keys()
        assert TmuxSession._tmux_send_keys is first


# ── Chunking correctness ─────────────────────────────────────────────────


class TestTmuxSendKeysByteChunking:
    def test_short_payload_is_single_command(self):
        session = _make_session()
        commands = session._tmux_send_keys(["echo hi", "Enter"])
        assert len(commands) == 1
        _assert_commands_below_limit(session, commands)

    def test_long_ascii_payload_split_below_limit(self):
        session = _make_session()
        long_key = "a" * (session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH * 2)

        commands = session._tmux_send_keys([long_key, "Enter"])

        assert len(commands) >= 2
        _assert_commands_below_limit(session, commands)
        all_payload = _extract_all_send_keys_payload(commands)
        assert "".join(all_payload[:-1]) == long_key
        assert all_payload[-1] == "Enter"

    def test_quote_heavy_payload_split_below_limit(self):
        session = _make_session()
        quote_heavy_key = "'" * (session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH // 2 + 100)

        commands = session._tmux_send_keys([quote_heavy_key, "Enter"])

        assert len(commands) >= 2
        _assert_commands_below_limit(session, commands)
        all_payload = _extract_all_send_keys_payload(commands)
        assert "".join(all_payload[:-1]) == quote_heavy_key
        assert all_payload[-1] == "Enter"

    def test_multibyte_payload_split_below_byte_limit(self):
        """A multibyte payload that fits by character count but not by byte count.

        ``"é"`` is 1 code point / 2 UTF-8 bytes. A payload of
        ``MAX // 2 + 100`` such chars has ~``MAX + 200`` bytes: the old
        character-count check would emit a single over-limit command, the
        byte-count fix must split it.
        """
        session = _make_session()
        multibyte_key = "é" * (session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH // 2 + 100)

        commands = session._tmux_send_keys([multibyte_key])

        assert len(commands) >= 2
        _assert_commands_below_limit(session, commands)
        assert "".join(_extract_all_send_keys_payload(commands)) == multibyte_key

    def test_many_small_keys_split_across_commands(self):
        session = _make_session()
        max_len = session._TMUX_SEND_KEYS_MAX_COMMAND_LENGTH
        keys = ["x" * 100] * (max_len // 100 + 10)

        commands = session._tmux_send_keys(keys)

        assert len(commands) >= 2
        _assert_commands_below_limit(session, commands)
        assert _extract_all_send_keys_payload(commands) == keys

    def test_lean4_proof_fixture_stays_below_byte_limit(self):
        session = _make_session()
        payload = LEAN4_PROOF_STEP33_FIXTURE.read_text()

        commands = session._tmux_send_keys([payload])

        _assert_commands_below_limit(session, commands)
        assert "".join(_extract_all_send_keys_payload(commands)) == payload


@pytest.mark.tmux
@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_lean4_proof_fixture_real_tmux_smoke():
    """Real tmux must accept every chunked ``send-keys`` command (no 'command too long')."""
    payload = LEAN4_PROOF_STEP33_FIXTURE.read_text()
    session_name = f"nel-test-{uuid.uuid4().hex}"
    session = _make_session(session_name=session_name)

    commands = session._tmux_send_keys([payload])
    _assert_commands_below_limit(session, commands)

    subprocess.run(
        ["tmux", "new-session", "-d", "-s", session_name],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        for command in commands:
            result = subprocess.run(command, shell=True, capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
    finally:
        subprocess.run(
            ["tmux", "kill-session", "-t", session_name],
            check=False,
            capture_output=True,
            text=True,
        )
