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
"""Tests for env_groups.py — per-group env var disambiguation."""

from nemo_evaluator.orchestration.secrets_env import (
    build_reexport_commands,
    generate_secrets_env,
    redact_secrets_env_content,
    reexport_keys,
)


class TestGenerateSecretsEnv:
    def test_empty_groups(self):
        result = generate_secrets_env({})
        assert result.secrets_content == ""
        assert result.group_remappings == {}

    def test_single_group_single_var(self):
        result = generate_secrets_env({"svc_model": {"HF_TOKEN": "hf_1234"}})
        assert "hf_1234" in result.secrets_content
        assert len(result.group_remappings["svc_model"]) == 1
        r = result.group_remappings["svc_model"][0]
        assert r.original_name == "HF_TOKEN"
        assert "HF_TOKEN_" in r.disambiguated_name
        assert "_SVC_MODEL" in r.disambiguated_name

    def test_different_groups_same_key(self):
        result = generate_secrets_env(
            {
                "svc_a": {"HF_TOKEN": "token_a"},
                "svc_b": {"HF_TOKEN": "token_b"},
            }
        )
        assert "token_a" in result.secrets_content
        assert "token_b" in result.secrets_content
        a_name = result.group_remappings["svc_a"][0].disambiguated_name
        b_name = result.group_remappings["svc_b"][0].disambiguated_name
        assert a_name != b_name

    def test_disambiguated_names_are_unique(self):
        result = generate_secrets_env(
            {
                "g1": {"X": "v1", "Y": "v2"},
                "g2": {"X": "v3", "Z": "v4"},
            }
        )
        all_names = set()
        for remappings in result.group_remappings.values():
            for r in remappings:
                all_names.add(r.disambiguated_name)
        assert len(all_names) == 4

    def test_escapes_double_quotes(self):
        result = generate_secrets_env({"g": {"K": 'val"with"quotes'}})
        assert r"val\"with\"quotes" in result.secrets_content

    def test_escapes_backslashes(self):
        result = generate_secrets_env({"g": {"K": r"path\to\thing"}})
        assert r"path\\to\\thing" in result.secrets_content


class TestBuildReexportCommands:
    def test_generates_exports(self):
        result = generate_secrets_env(
            {
                "svc_x": {"HF_TOKEN": "tok", "AWS_KEY": "key"},
            }
        )
        cmds = build_reexport_commands("svc_x", result)
        assert 'export HF_TOKEN="$' in cmds
        assert 'export AWS_KEY="$' in cmds

    def test_empty_for_unknown_group(self):
        result = generate_secrets_env({"svc_x": {"A": "1"}})
        cmds = build_reexport_commands("nonexistent", result)
        assert cmds == ""


class TestReexportKeys:
    def test_returns_original_names(self):
        result = generate_secrets_env(
            {
                "g": {"HF_TOKEN": "v1", "AWS_KEY": "v2"},
            }
        )
        keys = reexport_keys("g", result)
        assert "HF_TOKEN" in keys
        assert "AWS_KEY" in keys

    def test_empty_for_unknown(self):
        result = generate_secrets_env({})
        assert reexport_keys("x", result) == []


class TestRedactSecretsEnvContent:
    def test_redacts_long_values(self):
        content = 'export K_abc_G="longsecretvalue"\n'
        redacted = redact_secrets_env_content(content)
        assert "longsecretvalue" not in redacted
        assert "***alue" in redacted

    def test_fully_masks_short_values(self):
        content = 'export K_abc_G="tiny"\n'
        redacted = redact_secrets_env_content(content)
        assert '"***"' in redacted

    def test_non_export_lines_unchanged(self):
        content = '# comment line\nexport K="secret"\n'
        redacted = redact_secrets_env_content(content)
        assert "# comment line" in redacted


class TestEndToEndScript:
    """Integration-like tests: generate secrets, build re-exports, verify script behavior."""

    def test_two_services_get_isolated_env(self):
        groups = {
            "svc_nemotron": {"HF_TOKEN": "model_token", "CACHE": "/cache/a"},
            "svc_judge": {"HF_TOKEN": "judge_token"},
            "eval_swebench": {"HF_TOKEN": "eval_token"},
        }
        result = generate_secrets_env(groups)

        nemotron_re = build_reexport_commands("svc_nemotron", result)
        judge_re = build_reexport_commands("svc_judge", result)
        eval_re = build_reexport_commands("eval_swebench", result)

        assert "model_token" not in nemotron_re
        assert "judge_token" not in judge_re
        assert "eval_token" not in eval_re

        assert "HF_TOKEN" in nemotron_re
        assert "CACHE" in nemotron_re
        assert "HF_TOKEN" in judge_re
        assert "HF_TOKEN" in eval_re

        assert nemotron_re != judge_re
        assert judge_re != eval_re
