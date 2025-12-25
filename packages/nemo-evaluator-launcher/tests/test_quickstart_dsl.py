# SPDX-License-Identifier: Apache-2.0
import types

import pytest

from nemo_evaluator_launcher.cli.quickstart import Cmd


def _make_stub_prompters(cmd: Cmd, exec_choice: str, dep_choice: str):
    """Create bound stub methods for prompting that drive the flow deterministically.

    - First choice prompt: execution_default -> returns exec_choice
    - Second choice prompt: deployment_default -> returns dep_choice
    - Other prompts: return provided default when available; otherwise, minimal valid values
    """
    choice_calls = {"exec": False, "dep": False}

    def stub_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
        # Order-dependent routing by prompt text
        if "Select execution platform" in prompt and not choice_calls["exec"]:
            choice_calls["exec"] = True
            return exec_choice
        if "Select deployment type" in prompt and not choice_calls["dep"]:
            choice_calls["dep"] = True
            return dep_choice
        # Otherwise accept default if provided
        if default is not None:
            return default
        # Fall back to first choice when no default
        return choices[0]

    def stub_str(prompt: str, default: str | None = None, required: bool = False) -> str:
        if default:
            return default
        # Provide minimal but valid strings for required inputs
        low = prompt.lower()
        if "hostname" in low:
            return "headnode.local"
        if "account" in low:
            return "acct"
        if "output_dir" in low:
            return "/tmp/nv-eval-out"
        if "image (docker image)" in low:
            return "nvcr.io/nvidia/some:tag"
        if "command (server start command)" in low:
            return "python -m server"
        if "api_key_name" in low:
            return "API_KEY"
        if "model_id" in low:
            return "meta/llama-3.1-8b-instruct"
        if "url" in low:
            return "https://integrate.api.nvidia.com/v1/chat/completions"
        return "value"

    def stub_int(prompt: str, default: int | None = None) -> int:
        return default if default is not None else 1

    # Bind stubs as methods on the instance
    def _pc(self, prompt: str, choices: list[str], default: str | None = None):
        return stub_choice(prompt, choices, default)

    def _ps(self, prompt: str, default: str | None = None, required: bool = False):
        return stub_str(prompt, default, required)

    def _pi(self, prompt: str, default: int | None = None):
        return stub_int(prompt, default)

    cmd._prompt_choice = types.MethodType(_pc, cmd)
    cmd._prompt_str = types.MethodType(_ps, cmd)
    cmd._prompt_int = types.MethodType(_pi, cmd)


@pytest.mark.parametrize("advanced", [False, True])
def test_quickstart_dsl_branches_smoke(advanced: bool):
    """Traverse the DSL across discovered execution and deployment types and render YAML.

    This is a smoke test ensuring that for each (exec, deployment) pair, the flow
    can be completed using defaults and that rendering succeeds. When `advanced`
    is True, we also enable interceptors to exercise those branches.
    """
    cmd = Cmd()

    exec_choices = cmd._list_execution_defaults() or ["local", "slurm/default", "lepton/default"]
    dep_choices = cmd._list_deployment_types() or ["none", "vllm", "nim", "sglang", "trtllm", "generic"]

    # Limit the cartesian product to reasonable size while still walking all deployments
    exec_subset = exec_choices[:3] if len(exec_choices) > 3 else exec_choices

    for dep_choice in dep_choices:
        for exec_choice in exec_subset:
            cmd2 = Cmd()
            _make_stub_prompters(cmd2, exec_choice, dep_choice)

            flow = cmd2._build_flow(exec_choices, dep_choices)
            answers = cmd2._run_flow(flow)

            if advanced:
                # Flip advanced flags on to cover interceptors branch
                answers.update(
                    {
                        "enable_caching": True,
                        "reuse_cached": True,
                        "cache_dir": "/results/cache",
                        "enable_request_logging": True,
                        "max_requests": 2,
                        "enable_payload_modifier": True,
                        "enable_thinking": True,
                        "thinking_budget": 10,
                    }
                )

            # Ensure mandatory fields exist for the selected deployment
            dep = dep_choice
            if dep == "none":
                assert "model_id" in answers and "url" in answers
            elif dep == "vllm":
                assert "served_model_name" in answers and "hf_model_handle" in answers
            elif dep == "sglang":
                assert "served_model_name" in answers and "hf_model_handle" in answers
            elif dep == "trtllm":
                assert "served_model_name" in answers and "checkpoint_path" in answers
            elif dep == "generic":
                assert "image" in answers and "command" in answers and "served_model_name" in answers
            elif dep == "nim":
                assert "served_model_name" in answers

            # Render YAML; ensure it contains defaults header and does not error
            yaml_text = cmd2._render_yaml(answers)
            assert isinstance(yaml_text, str)
            assert "defaults:" in yaml_text
            assert f"deployment: {dep_choice}" in yaml_text or dep_choice == "none"

            # If advanced enabled, YAML should include interceptors
            if advanced:
                assert "interceptors:" in yaml_text
