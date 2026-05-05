# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for nemo_evaluator.templates — resolution, rendering, and autoescape safety."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_evaluator.templates import (
    _TEMPLATES_DIR,
    _jinja_env,
    render_template,
    resolve_template_path,
)

# ── resolve_template_path ─────────────────────────────────────────────────


class TestResolveTemplatePath:
    def test_none_returns_none(self):
        assert resolve_template_path(None) is None

    @pytest.mark.parametrize("value", ["off", "OFF", "none", "None", ""])
    def test_disabled_sentinels(self, value: str):
        assert resolve_template_path(value) is None

    def test_builtin_filename(self):
        path = resolve_template_path("swebench-instruction.md")
        assert path is not None
        assert path.exists()
        assert path.parent == _TEMPLATES_DIR

    def test_absolute_path(self, tmp_path: Path):
        tmpl = tmp_path / "custom.md"
        tmpl.write_text("{{ prompt }}")
        result = resolve_template_path(str(tmpl))
        assert result == tmpl

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="instruction_template not found"):
            resolve_template_path(str(tmp_path / "nonexistent.md"))


# ── render_template ───────────────────────────────────────────────────────


class TestRenderTemplate:
    def test_basic_variable_substitution(self, tmp_path: Path):
        tmpl = tmp_path / "simple.md"
        tmpl.write_text("Hello {{ workspace_path }}, prompt={{ original_prompt }}")

        result = render_template(
            tmpl,
            original_prompt="solve it",
            workspace_path="/testbed",
            metadata={},
        )
        assert result == "Hello /testbed, prompt=solve it"

    def test_metadata_keys_available_directly(self, tmp_path: Path):
        tmpl = tmp_path / "meta.md"
        tmpl.write_text("repo={{ repo }}, commit={{ base_commit }}")

        result = render_template(
            tmpl,
            original_prompt="fix bug",
            workspace_path="/w",
            metadata={"repo": "my-repo", "base_commit": "deadbeef"},
        )
        assert result == "repo=my-repo, commit=deadbeef"

    def test_instance_dict_contains_problem_statement(self, tmp_path: Path):
        tmpl = tmp_path / "inst.md"
        tmpl.write_text("stmt={{ instance.problem_statement }}")

        result = render_template(
            tmpl,
            original_prompt="the original prompt",
            workspace_path="/w",
            metadata={},
        )
        assert result == "stmt=the original prompt"

    def test_swebench_builtin_renders(self):
        path = _TEMPLATES_DIR / "swebench-instruction.md"
        result = render_template(
            path,
            original_prompt="Something is broken",
            workspace_path="/testbed",
            metadata={"base_commit": "abc123"},
        )
        assert "/testbed" in result
        assert "Something is broken" in result
        assert "abc123" in result

    def test_output_is_stripped(self, tmp_path: Path):
        tmpl = tmp_path / "pad.md"
        tmpl.write_text("\n\n  hello  \n\n")

        result = render_template(tmpl, original_prompt="", workspace_path="", metadata={})
        assert result == "hello"


# ── autoescape safety ─────────────────────────────────────────────────────


class TestAutoescapeSafety:
    """Verify that HTML-like content is never escaped in rendered output.

    Templates produce plain-text LLM prompts, not HTML.  Characters like
    ``<``, ``>``, and ``&`` must stay literal.
    """

    def test_angle_brackets_preserved(self, tmp_path: Path):
        tmpl = tmp_path / "tags.md"
        tmpl.write_text("<uploaded_files>{{ workspace_path }}</uploaded_files>")

        result = render_template(tmpl, original_prompt="", workspace_path="/app", metadata={})
        assert "<uploaded_files>/app</uploaded_files>" == result
        assert "&lt;" not in result

    def test_ampersand_preserved(self, tmp_path: Path):
        tmpl = tmp_path / "amp.md"
        tmpl.write_text("{{ original_prompt }}")

        result = render_template(
            tmpl,
            original_prompt="A & B < C",
            workspace_path="",
            metadata={},
        )
        assert result == "A & B < C"
        assert "&amp;" not in result
        assert "&lt;" not in result

    def test_autoescape_is_callable(self):
        assert callable(_jinja_env.autoescape)


# ── ByobInstalledAgent Jinja rendering ────────────────────────────────────


class TestByobAgentTemplates:
    """Verify that ByobInstalledAgent renders shell scripts without escaping."""

    def _make_agent(self, tmp_path: Path, install: str | None = None, run: str = "echo ok"):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()
        (agent_dir / "agent.toml").write_text('[agent]\nname = "test"\ntimeout_sec = 60\n')
        if install is not None:
            (agent_dir / "install.sh.j2").write_text(install)
        (agent_dir / "run.sh.j2").write_text(run)

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        from nemo_evaluator.solvers.byob_agent import ByobInstalledAgent

        return ByobInstalledAgent(agent_dir=agent_dir, logs_dir=logs_dir, model="gpt-test")

    @pytest.mark.asyncio
    async def test_run_preserves_special_chars(self, tmp_path: Path):
        agent = self._make_agent(
            tmp_path,
            run='echo "{{ instruction }}" && echo {{ model }}',
        )

        env_mock = AsyncMock()
        env_mock.exec.return_value = MagicMock(return_code=0, stdout="done", stderr="")

        await agent.run(
            instruction="Fix <bug> in module & class",
            environment=env_mock,
            context=None,
        )

        rendered = env_mock.exec.call_args.kwargs["command"]
        assert "<bug>" in rendered, "angle brackets were escaped"
        assert "& class" in rendered, "ampersand was escaped"
        assert "&lt;" not in rendered
        assert "&amp;" not in rendered
        assert "gpt-test" in rendered

    @pytest.mark.asyncio
    async def test_setup_renders_install_script(self, tmp_path: Path):
        agent = self._make_agent(
            tmp_path,
            install="pip install {{ model }}",
        )

        env_mock = AsyncMock()
        env_mock.exec.return_value = MagicMock(return_code=0, stderr="")

        await agent.setup(env_mock)

        script = (tmp_path / "logs" / "install.sh").read_text()
        assert script == "pip install gpt-test"

    @pytest.mark.asyncio
    async def test_setup_skipped_when_no_install_template(self, tmp_path: Path):
        agent = self._make_agent(tmp_path, install=None)

        env_mock = AsyncMock()
        await agent.setup(env_mock)
        env_mock.exec.assert_not_called()

    @pytest.mark.asyncio
    async def test_setup_raises_on_nonzero_exit(self, tmp_path: Path):
        agent = self._make_agent(tmp_path, install="exit 1")

        env_mock = AsyncMock()
        env_mock.exec.return_value = MagicMock(return_code=1, stderr="fail")

        with pytest.raises(RuntimeError, match="BYOB agent setup failed"):
            await agent.setup(env_mock)

    def test_missing_run_template_raises(self, tmp_path: Path):
        agent_dir = tmp_path / "agent"
        agent_dir.mkdir()

        logs_dir = tmp_path / "logs"
        logs_dir.mkdir()

        from nemo_evaluator.solvers.byob_agent import ByobInstalledAgent

        agent = ByobInstalledAgent(agent_dir=agent_dir, logs_dir=logs_dir)
        with pytest.raises(FileNotFoundError, match="run.sh.j2"):
            _ = agent._run_template
