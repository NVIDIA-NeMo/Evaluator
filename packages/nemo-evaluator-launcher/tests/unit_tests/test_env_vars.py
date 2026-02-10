# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for nemo_evaluator_launcher.common.env_vars module."""

import os
import warnings
from unittest import mock

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.env_vars import (
    EnvVarFromHost,
    EnvVarLiteral,
    EnvVarRuntime,
    SecretsEnvResult,
    VarRemapping,
    build_reexport_commands,
    collect_deployment_env_vars,
    collect_eval_env_vars,
    generate_secrets_env,
    parse_env_var_value,
    redact_secrets_env_content,
    resolve_env_var,
)

# --- parse_env_var_value ---


class TestParseEnvVarValue:
    def test_literal_prefix(self):
        result = parse_env_var_value("$lit:some/path/here")
        assert result == EnvVarLiteral(value="some/path/here")

    def test_literal_empty_value(self):
        result = parse_env_var_value("$lit:")
        assert result == EnvVarLiteral(value="")

    def test_host_prefix(self):
        result = parse_env_var_value("$host:HF_TOKEN")
        assert result == EnvVarFromHost(host_var_name="HF_TOKEN")

    def test_runtime_prefix(self):
        result = parse_env_var_value("$runtime:SLURM_JOB_ID")
        assert result == EnvVarRuntime(runtime_var_name="SLURM_JOB_ID")

    def test_hydra_resolver_raises_error(self):
        with pytest.raises(ValueError, match="Hydra resolver syntax"):
            parse_env_var_value("${oc.env:HF_TOKEN}")

    def test_hydra_decode_resolver_raises_error(self):
        with pytest.raises(ValueError, match="Hydra resolver syntax"):
            parse_env_var_value("${oc.decode:something}")

    def test_backward_compat_dollar_prefix(self):
        with pytest.warns(DeprecationWarning, match="Use '\\$host:HF_TOKEN'"):
            result = parse_env_var_value("$HF_TOKEN")
        assert result == EnvVarFromHost(host_var_name="HF_TOKEN")

    def test_backward_compat_bare_name(self):
        with pytest.warns(DeprecationWarning, match="Use '\\$host:MY_VAR'"):
            result = parse_env_var_value("MY_VAR")
        assert result == EnvVarFromHost(host_var_name="MY_VAR")

    def test_backward_compat_path_value(self):
        with pytest.warns(DeprecationWarning, match="Use '\\$lit:/some/path'"):
            result = parse_env_var_value("/some/path")
        assert result == EnvVarLiteral(value="/some/path")

    def test_backward_compat_url_value(self):
        with pytest.warns(DeprecationWarning, match="Use '\\$lit:"):
            result = parse_env_var_value("http://example.com")
        assert result == EnvVarLiteral(value="http://example.com")

    def test_backward_compat_value_with_spaces(self):
        with pytest.warns(DeprecationWarning, match="Use '\\$lit:"):
            result = parse_env_var_value("hello world")
        assert result == EnvVarLiteral(value="hello world")


# --- resolve_env_var ---


class TestResolveEnvVar:
    def test_literal_resolves_to_value(self):
        name, value = resolve_env_var("MY_VAR", EnvVarLiteral(value="/some/path"))
        assert name == "MY_VAR"
        assert value == "/some/path"

    def test_host_resolves_from_env(self):
        with mock.patch.dict(os.environ, {"SOURCE_VAR": "secret_value"}):
            name, value = resolve_env_var(
                "TARGET_VAR", EnvVarFromHost(host_var_name="SOURCE_VAR")
            )
        assert name == "TARGET_VAR"
        assert value == "secret_value"

    def test_host_raises_if_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="is not set"):
                resolve_env_var(
                    "TARGET_VAR", EnvVarFromHost(host_var_name="MISSING_VAR")
                )

    def test_host_empty_string_is_valid(self):
        with mock.patch.dict(os.environ, {"EMPTY_VAR": ""}):
            name, value = resolve_env_var(
                "TARGET", EnvVarFromHost(host_var_name="EMPTY_VAR")
            )
        assert value == ""

    def test_runtime_resolves_to_none(self):
        name, value = resolve_env_var(
            "MY_VAR", EnvVarRuntime(runtime_var_name="RUNTIME_VAR")
        )
        assert name == "MY_VAR"
        assert value is None


# --- generate_secrets_env ---


class TestGenerateSecretsEnv:
    def test_single_group_literal(self):
        result = generate_secrets_env(
            {"deployment": {"HF_HOME": EnvVarLiteral(value="/hf-cache")}}
        )
        assert "export HF_HOME_" in result.secrets_content
        assert "=/hf-cache" in result.secrets_content
        assert len(result.group_remappings["deployment"]) == 1
        assert result.group_remappings["deployment"][0].original_name == "HF_HOME"

    def test_single_group_host(self):
        with mock.patch.dict(os.environ, {"MY_TOKEN": "secret123"}):
            result = generate_secrets_env(
                {"eval_task": {"TOKEN": EnvVarFromHost(host_var_name="MY_TOKEN")}}
            )
        assert "=secret123" in result.secrets_content
        assert len(result.group_remappings["eval_task"]) == 1

    def test_runtime_var_not_in_secrets_content(self):
        result = generate_secrets_env(
            {"deployment": {"RUNTIME_VAR": EnvVarRuntime(runtime_var_name="JOB_ID")}}
        )
        assert result.secrets_content == ""
        assert len(result.runtime_vars["deployment"]) == 1
        assert result.runtime_vars["deployment"][0].original_name == "RUNTIME_VAR"
        # Runtime vars reference the runtime_var_name directly, not a disambiguated suffix
        assert result.runtime_vars["deployment"][0].disambiguated_name == "JOB_ID"

    def test_multiple_groups_disambiguation(self):
        with mock.patch.dict(os.environ, {"SRC_A": "val_a", "SRC_B": "val_b"}):
            result = generate_secrets_env(
                {
                    "task_a": {"HF_TOKEN": EnvVarFromHost(host_var_name="SRC_A")},
                    "task_b": {"HF_TOKEN": EnvVarFromHost(host_var_name="SRC_B")},
                }
            )
        # Both should be in secrets content but disambiguated
        assert "=val_a" in result.secrets_content
        assert "=val_b" in result.secrets_content
        # Different disambiguated names
        name_a = result.group_remappings["task_a"][0].disambiguated_name
        name_b = result.group_remappings["task_b"][0].disambiguated_name
        assert name_a != name_b
        assert "TASK_A" in name_a
        assert "TASK_B" in name_b

    def test_empty_groups(self):
        result = generate_secrets_env({})
        assert result.secrets_content == ""
        assert result.group_remappings == {}
        assert result.runtime_vars == {}

    def test_mixed_types_in_group(self):
        with mock.patch.dict(os.environ, {"HOST_VAR": "host_val"}):
            result = generate_secrets_env(
                {
                    "my_group": {
                        "LIT_VAR": EnvVarLiteral(value="literal"),
                        "HOST_VAR": EnvVarFromHost(host_var_name="HOST_VAR"),
                        "RT_VAR": EnvVarRuntime(runtime_var_name="RT"),
                    }
                }
            )
        assert len(result.group_remappings["my_group"]) == 2  # lit + host
        assert len(result.runtime_vars["my_group"]) == 1  # runtime
        assert "=literal" in result.secrets_content
        assert "=host_val" in result.secrets_content

    def test_special_chars_in_group_name(self):
        result = generate_secrets_env(
            {"my-task.123": {"VAR": EnvVarLiteral(value="x")}}
        )
        disambiguated = result.group_remappings["my-task.123"][0].disambiguated_name
        # Special chars should be replaced with underscores
        assert "MY_TASK_123" in disambiguated


# --- build_reexport_commands ---


class TestBuildReexportCommands:
    def test_basic_reexport(self):
        result = SecretsEnvResult(
            secrets_content="",
            group_remappings={
                "task_a": [
                    VarRemapping(
                        original_name="HF_TOKEN",
                        disambiguated_name="HF_TOKEN_abc_TASK_A",
                    )
                ]
            },
        )
        cmd = build_reexport_commands("task_a", result)
        assert cmd == "export HF_TOKEN=$HF_TOKEN_abc_TASK_A"

    def test_multiple_vars(self):
        result = SecretsEnvResult(
            secrets_content="",
            group_remappings={
                "task_a": [
                    VarRemapping("HF_TOKEN", "HF_TOKEN_abc_TASK_A"),
                    VarRemapping("API_KEY", "API_KEY_abc_TASK_A"),
                ]
            },
        )
        cmd = build_reexport_commands("task_a", result)
        assert "export HF_TOKEN=$HF_TOKEN_abc_TASK_A" in cmd
        assert "export API_KEY=$API_KEY_abc_TASK_A" in cmd
        assert " ; " in cmd

    def test_empty_group(self):
        result = SecretsEnvResult(secrets_content="")
        cmd = build_reexport_commands("nonexistent", result)
        assert cmd == ""

    def test_includes_runtime_vars(self):
        # Runtime vars reference the runtime_var_name directly (e.g. NGC_API_TOKEN),
        # not a disambiguated suffix, because the value comes from the environment.
        result = SecretsEnvResult(
            secrets_content="",
            group_remappings={
                "task_a": [VarRemapping("HF_TOKEN", "HF_TOKEN_abc_TASK_A")]
            },
            runtime_vars={"task_a": [VarRemapping("API_KEY", "NGC_API_TOKEN")]},
        )
        cmd = build_reexport_commands("task_a", result)
        assert "export HF_TOKEN=$HF_TOKEN_abc_TASK_A" in cmd
        assert "export API_KEY=$NGC_API_TOKEN" in cmd


# --- Config collection with hierarchical merging ---


class TestCollectEvalEnvVars:
    """Tests for collect_eval_env_vars with hierarchical merging."""

    def _make_cfg(self, overrides=None):
        base = {
            "env_vars": {},
            "evaluation": {"env_vars": {}, "tasks": []},
            "execution": {"env_vars": {}},
            "deployment": {"type": "none"},
            "target": {"api_endpoint": {}},
        }
        if overrides:
            # Deep merge
            for key, val in overrides.items():
                if (
                    isinstance(val, dict)
                    and key in base
                    and isinstance(base[key], dict)
                ):
                    base[key].update(val)
                else:
                    base[key] = val
        return OmegaConf.create(base)

    def test_top_level_env_vars_flow_to_eval(self):
        """Top-level env_vars are included in eval collection."""
        cfg = self._make_cfg({"env_vars": {"HF_TOKEN": "$host:HF_TOKEN"}})
        task = OmegaConf.create({"name": "test_task", "env_vars": {}})
        task_def = {}

        result = collect_eval_env_vars(cfg, task, task_def)
        assert "HF_TOKEN" in result
        assert result["HF_TOKEN"] == EnvVarFromHost(host_var_name="HF_TOKEN")

    def test_eval_level_overrides_top_level(self):
        """evaluation.env_vars overrides top-level env_vars."""
        cfg = self._make_cfg(
            {
                "env_vars": {"HF_TOKEN": "$host:GLOBAL_TOKEN"},
                "evaluation": {"env_vars": {"HF_TOKEN": "$host:EVAL_TOKEN"}},
            }
        )
        task = OmegaConf.create({"name": "test_task", "env_vars": {}})
        task_def = {}

        result = collect_eval_env_vars(cfg, task, task_def)
        assert result["HF_TOKEN"] == EnvVarFromHost(host_var_name="EVAL_TOKEN")

    def test_task_level_overrides_eval_level(self):
        """task.env_vars overrides evaluation.env_vars."""
        cfg = self._make_cfg(
            {
                "evaluation": {"env_vars": {"HF_TOKEN": "$host:EVAL_TOKEN"}},
            }
        )
        task = OmegaConf.create(
            {
                "name": "test_task",
                "env_vars": {"HF_TOKEN": "$host:TASK_TOKEN"},
            }
        )
        task_def = {}

        result = collect_eval_env_vars(cfg, task, task_def)
        assert result["HF_TOKEN"] == EnvVarFromHost(host_var_name="TASK_TOKEN")

    def test_full_hierarchy_last_wins(self):
        """Full hierarchy: top-level → eval → task (last wins)."""
        cfg = self._make_cfg(
            {
                "env_vars": {
                    "SHARED": "$lit:top",
                    "TOP_ONLY": "$lit:top_only",
                },
                "evaluation": {
                    "env_vars": {
                        "SHARED": "$lit:eval",
                        "EVAL_ONLY": "$lit:eval_only",
                    },
                },
            }
        )
        task = OmegaConf.create(
            {
                "name": "test_task",
                "env_vars": {"SHARED": "$lit:task"},
            }
        )
        task_def = {}

        result = collect_eval_env_vars(cfg, task, task_def)
        assert result["SHARED"] == EnvVarLiteral(value="task")
        assert result["TOP_ONLY"] == EnvVarLiteral(value="top_only")
        assert result["EVAL_ONLY"] == EnvVarLiteral(value="eval_only")

    def test_deprecated_execution_eval_env_vars_merged(self):
        """execution.env_vars.evaluation is merged with deprecation warning."""
        cfg = self._make_cfg(
            {
                "execution": {
                    "env_vars": {"evaluation": {"EXEC_VAR": "$lit:exec_val"}}
                },
            }
        )
        task = OmegaConf.create({"name": "test_task", "env_vars": {}})
        task_def = {}

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collect_eval_env_vars(cfg, task, task_def)

        assert result["EXEC_VAR"] == EnvVarLiteral(value="exec_val")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any(
            "execution.env_vars.evaluation" in str(x.message) for x in dep_warnings
        )

    def test_deprecated_execution_eval_overrides_earlier(self):
        """execution.env_vars.evaluation (deprecated) overrides top-level and eval."""
        cfg = self._make_cfg(
            {
                "env_vars": {"VAR": "$lit:top"},
                "evaluation": {"env_vars": {"VAR": "$lit:eval"}},
                "execution": {"env_vars": {"evaluation": {"VAR": "$lit:exec"}}},
            }
        )
        task = OmegaConf.create({"name": "test_task", "env_vars": {}})
        task_def = {}

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = collect_eval_env_vars(cfg, task, task_def)

        # Deprecated exec path overrides eval-level (it merges after)
        assert result["VAR"] == EnvVarLiteral(value="exec")


class TestCollectDeploymentEnvVars:
    """Tests for collect_deployment_env_vars with hierarchical merging."""

    def _make_cfg(self, overrides=None):
        base = {
            "env_vars": {},
            "evaluation": {},
            "execution": {"env_vars": {}},
            "deployment": {"type": "none"},
        }
        if overrides:
            for key, val in overrides.items():
                if (
                    isinstance(val, dict)
                    and key in base
                    and isinstance(base[key], dict)
                ):
                    base[key].update(val)
                else:
                    base[key] = val
        return OmegaConf.create(base)

    def test_top_level_env_vars_flow_to_deployment(self):
        """Top-level env_vars are included in deployment collection."""
        cfg = self._make_cfg({"env_vars": {"SHARED_VAR": "$lit:shared"}})

        result = collect_deployment_env_vars(cfg)
        assert result["SHARED_VAR"] == EnvVarLiteral(value="shared")

    def test_deprecated_exec_deployment_overrides_top_level(self):
        """execution.env_vars.deployment overrides top-level."""
        cfg = self._make_cfg(
            {
                "env_vars": {"VAR": "$lit:top"},
                "execution": {"env_vars": {"deployment": {"VAR": "$lit:exec"}}},
            }
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = collect_deployment_env_vars(cfg)

        assert result["VAR"] == EnvVarLiteral(value="exec")
        dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
        assert any(
            "execution.env_vars.deployment" in str(x.message) for x in dep_warnings
        )

    def test_deprecated_deployment_env_vars_overrides_all(self):
        """cfg.deployment.env_vars overrides everything (last wins)."""
        cfg = self._make_cfg(
            {
                "env_vars": {"VAR": "$lit:top"},
                "execution": {"env_vars": {"deployment": {"VAR": "$lit:exec"}}},
                "deployment": {"type": "none", "env_vars": {"VAR": "$lit:deploy"}},
            }
        )

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = collect_deployment_env_vars(cfg)

        assert result["VAR"] == EnvVarLiteral(value="deploy")

    def test_no_env_vars_returns_empty(self):
        """No env vars defined anywhere returns empty dict."""
        cfg = self._make_cfg()
        result = collect_deployment_env_vars(cfg)
        assert result == {}


# --- redact_secrets_env_content ---


class TestRedactSecretsEnvContent:
    def test_long_value_shows_last_four(self):
        content = "export MY_KEY=super_secret_value\n"
        result = redact_secrets_env_content(content)
        assert result == "export MY_KEY=***alue\n"

    def test_short_value_fully_masked(self):
        content = "export KEY=abcd\n"
        result = redact_secrets_env_content(content)
        assert result == "export KEY=***\n"

    def test_very_short_value_fully_masked(self):
        content = "export KEY=ab\n"
        result = redact_secrets_env_content(content)
        assert result == "export KEY=***\n"

    def test_empty_value(self):
        content = "export KEY=\n"
        result = redact_secrets_env_content(content)
        assert result == "export KEY=***\n"

    def test_multiple_lines(self):
        content = "export A=long_secret_a\nexport B=long_secret_b\n"
        result = redact_secrets_env_content(content)
        assert "export A=***et_a" in result
        assert "export B=***et_b" in result

    def test_non_export_lines_pass_through(self):
        content = "# comment\nexport KEY=secret_val\n"
        result = redact_secrets_env_content(content)
        assert result.startswith("# comment\n")
        assert "export KEY=***_val" in result

    def test_empty_content(self):
        result = redact_secrets_env_content("")
        assert result == ""

    def test_value_with_equals_sign(self):
        content = "export KEY=val=ue=with=equals\n"
        result = redact_secrets_env_content(content)
        assert result == "export KEY=***uals\n"

    def test_literal_keys_shown_unredacted(self):
        content = "export SECRET_abc_TASK=my_secret_key\nexport PATH_abc_TASK=/some/long/path\n"
        result = redact_secrets_env_content(
            content, literal_disambiguated_names={"PATH_abc_TASK"}
        )
        assert "export SECRET_abc_TASK=***_key" in result
        assert "export PATH_abc_TASK=/some/long/path" in result

    def test_all_literal_keys_unredacted(self):
        content = "export A=value_a\nexport B=value_b\n"
        result = redact_secrets_env_content(
            content, literal_disambiguated_names={"A", "B"}
        )
        assert result == "export A=value_a\nexport B=value_b\n"

    def test_no_literal_keys_all_redacted(self):
        content = "export A=value_a\nexport B=value_b\n"
        result = redact_secrets_env_content(content, literal_disambiguated_names=set())
        assert "export A=***ue_a" in result
        assert "export B=***ue_b" in result
