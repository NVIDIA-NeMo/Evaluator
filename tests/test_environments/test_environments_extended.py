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
"""Tests for evaluation environments — Harbor, Skills, LmEval, Composite, Container."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

# ── CompositeEnvironment ─────────────────────────────────────────────────


class TestCompositeEnvironment:
    def _make_envs(self):
        seed_env = MagicMock(spec=EvalEnvironment)
        verify_env = MagicMock(spec=EvalEnvironment)
        seed_env.dataset_size = AsyncMock(return_value=3)
        seed_env.seed = AsyncMock(return_value=SeedResult(prompt="q", expected_answer="a", metadata={"idx": 0}))
        verify_env.verify = AsyncMock(return_value=VerifyResult(reward=1.0, extracted_answer="a"))
        seed_env.close = AsyncMock()
        verify_env.close = AsyncMock()
        return seed_env, verify_env

    async def test_seed_delegates(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        result = await comp.seed(0)
        seed_env.seed.assert_called_once_with(0)
        assert result.prompt == "q"

    async def test_verify_delegates(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        result = await comp.verify("a", "a")
        verify_env.verify.assert_called_once()
        assert result.reward == 1.0

    async def test_close_both(self):
        from nemo_evaluator.environments.composite import CompositeEnvironment

        seed_env, verify_env = self._make_envs()
        comp = CompositeEnvironment(seed_env, verify_env)
        await comp.close()
        seed_env.close.assert_called_once()
        verify_env.close.assert_called_once()


# ── ContainerEnvironment ────────────────────────────────────────────────


class TestContainerEnvironment:
    @pytest.mark.parametrize("method,args", [("seed", (0,)), ("verify", ("a", "b"))], ids=["seed", "verify"])
    def test_seed_and_verify_raise_not_implemented(self, method, args):
        """ContainerEnvironment uses ``run_batch()``; the seed/verify cycle isn't supported."""
        import asyncio

        from nemo_evaluator.environments.container import ContainerEnvironment

        env = ContainerEnvironment(image="test:latest")
        with pytest.raises(NotImplementedError):
            asyncio.run(getattr(env, method)(*args))

    async def test_run_batch_end_to_end(self):
        """Full run_batch flow: config -> docker command -> results parsing."""
        import yaml as _yaml

        from nemo_evaluator.environments.container import ContainerEnvironment

        # Realistic v1 results.yml (matches EvaluationResult schema)
        v1_results = {
            "results": {
                "groups": {
                    "mmlu": {
                        "metrics": {
                            "pass@1": {
                                "scores": {
                                    "symbolic_correct": {"value": 80.0, "stats": {"count": 50}},
                                }
                            }
                        }
                    }
                }
            }
        }

        captured_cmd = []
        captured_run_config = {}

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(args)
            for arg in args:
                if not isinstance(arg, str):
                    continue
                if arg.endswith(":/results"):
                    Path(arg.split(":")[0]).joinpath("results.yml").write_text(_yaml.dump(v1_results))
                if arg.endswith(":/config/run_config.yaml:ro"):
                    captured_run_config.update(_yaml.safe_load(Path(arg.split(":")[0]).read_text()))
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        env = ContainerEnvironment(image="test:latest", task="ns_mmlu")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await env.run_batch(
                config={
                    "base_url": "https://api.example.com/v1/chat/completions",
                    "model": "test-model",
                    "api_key": "test-key",
                    "endpoint_type": "chat",
                    "params": {"limit_samples": 50},
                }
            )

        # Verify docker command structure
        cmd_str = " ".join(str(c) for c in captured_cmd)
        assert "docker run --rm" in cmd_str
        assert "test:latest" in cmd_str
        assert "NEMO_API_KEY=test-key" in cmd_str
        # Entrypoint: nemo-evaluator with eval-factory fallback
        assert "command -v nemo-evaluator" in cmd_str
        assert "eval-factory" in cmd_str
        assert "run_eval --run_config" in cmd_str

        # Verify generated run_config.yaml matches v1 schema
        assert captured_run_config["config"]["type"] == "ns_mmlu"
        assert captured_run_config["config"]["output_dir"] == "/results"
        assert captured_run_config["config"]["params"]["limit_samples"] == 50
        assert captured_run_config["target"]["api_endpoint"]["url"] == "https://api.example.com/v1/chat/completions"
        assert captured_run_config["target"]["api_endpoint"]["model_id"] == "test-model"
        assert captured_run_config["target"]["api_endpoint"]["api_key_name"] == "NEMO_API_KEY"
        assert captured_run_config["target"]["api_endpoint"]["type"] == "chat"

        # Verify results parsed correctly from v1 format
        assert result["benchmark"]["name"] == "container/ns_mmlu"
        assert result["benchmark"]["samples"] == 50
        assert result["benchmark"]["scores"]["mmlu/pass@1/symbolic_correct"]["value"] == 80.0

    def test_adapter_config_written_verbatim(self):
        """adapter_config lands at target.api_endpoint.adapter_config verbatim."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        adapter_cfg = {
            "use_reasoning": True,
            "params_to_add": {"chat_template_kwargs": {"enable_thinking": True}},
            "params_to_remove": ["max_tokens", "max_completion_tokens"],
        }
        env = ContainerEnvironment(image="test:latest", task="ns_mmlu")
        rc = env._build_legacy_run_config("http://m/v1", "m", "chat", {"temperature": 1.0}, adapter_cfg)
        assert rc["target"]["api_endpoint"]["adapter_config"] == adapter_cfg
        assert "adapter_config" not in rc["config"].get("params", {})

    async def test_run_batch_pyxis_dispatch_on_slurm(self, monkeypatch):
        """When SLURM_JOB_ID is set, dispatch via `srun --container-image`."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        # ``shutil.which`` should find srun for the dispatch to proceed.
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun" if name == "srun" else None)

        captured_cmd: list[str] = []
        captured_env: dict = {}

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(a for a in args if isinstance(a, str))
            if kwargs.get("env") is not None:
                captured_env.update(kwargs["env"])
            from pathlib import Path

            # mounts flag holds all mounts; results dir is in there
            for a in args:
                if isinstance(a, str) and a.startswith("--container-mounts="):
                    for entry in a.split("=", 1)[1].split(","):
                        if entry.endswith(":/results"):
                            Path(entry.split(":")[0]).joinpath("results.yml").write_text("results: {tasks: {}}\n")
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        # Use a sqsh-style file path to exercise the file-vs-registry detection.
        env = ContainerEnvironment(image="/srv/nemo-skills.sqsh", task="ns_mmlu")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await env.run_batch(
                config={
                    "base_url": "http://m/v1",
                    "model": "m",
                    "api_key": "k",
                    "endpoint_type": "chat",
                    "env_vars": {"JUDGE_API_KEY": "judge-secret"},
                    "mounts": {"/host/data": "/data"},
                }
            )

        # Pyxis-shaped argv, not docker
        assert "srun" in captured_cmd
        assert "docker" not in captured_cmd
        # Single-node, overlap, unbuffered
        assert "--mpi=pmix" in captured_cmd
        assert "--overlap" in captured_cmd
        assert "--unbuffered" in captured_cmd
        assert "--nodes=1" in captured_cmd
        assert "--ntasks=1" in captured_cmd
        assert "--no-container-mount-home" in captured_cmd
        # File-path image passes through (no docker:// prefix)
        assert "--container-image=/srv/nemo-skills.sqsh" in captured_cmd
        # Mounts joined into one comma-separated flag
        mount_flag = next(c for c in captured_cmd if c.startswith("--container-mounts="))
        assert "/host/data:/data" in mount_flag
        assert ":/results" in mount_flag
        assert ":/config/run_config.yaml:ro" in mount_flag
        # Env *names* on argv; *values* in subprocess env (Pyxis inherits)
        assert "--container-env=JUDGE_API_KEY" in captured_cmd
        assert "--container-env=NEMO_API_KEY" in captured_cmd
        assert "JUDGE_API_KEY=judge-secret" not in " ".join(captured_cmd)  # value not in argv
        assert captured_env.get("JUDGE_API_KEY") == "judge-secret"
        assert captured_env.get("NEMO_API_KEY") == "k"

    async def test_run_batch_pyxis_registry_uri_gets_docker_prefix(self, monkeypatch):
        """Registry URIs (no leading slash) get the ``docker://`` prefix for Pyxis."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/srun" if name == "srun" else None)

        captured_cmd: list[str] = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(a for a in args if isinstance(a, str))
            from pathlib import Path

            for a in args:
                if isinstance(a, str) and a.startswith("--container-mounts="):
                    for entry in a.split("=", 1)[1].split(","):
                        if entry.endswith(":/results"):
                            Path(entry.split(":")[0]).joinpath("results.yml").write_text("results: {tasks: {}}\n")
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        env = ContainerEnvironment(image="registry.example.com:5005/x:tag", task="ns_mmlu")
        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await env.run_batch(config={"base_url": "http://m", "model": "m"})

        assert "--container-image=docker://registry.example.com:5005/x:tag" in captured_cmd

    async def test_run_batch_slurm_without_srun_raises_clear_error(self, monkeypatch):
        """If SLURM_JOB_ID is set but srun isn't installed, surface a helpful error."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        monkeypatch.setenv("SLURM_JOB_ID", "12345")
        monkeypatch.setattr("shutil.which", lambda name: None)

        env = ContainerEnvironment(image="test:latest", task="ns_mmlu")
        with pytest.raises(RuntimeError, match=r"srun.*not in PATH|unset SLURM_JOB_ID"):
            await env.run_batch(config={"base_url": "http://m", "model": "m"})

    def test_redact_cmd_for_log_hides_env_values(self):
        """Secrets in -e K=V must not leak into INFO logs."""
        from nemo_evaluator.environments.container import _redact_cmd_for_log

        cmd = [
            "docker",
            "run",
            "--rm",
            "-v",
            "/tmp/x:/results",
            "-e",
            "NEMO_API_KEY=nvapi-SECRETSECRET",
            "-e",
            "JUDGE_API_KEY=another-secret",
            "image:tag",
        ]
        out = _redact_cmd_for_log(cmd)
        assert "nvapi-SECRETSECRET" not in out
        assert "another-secret" not in out
        assert "NEMO_API_KEY=<REDACTED>" in out
        assert "JUDGE_API_KEY=<REDACTED>" in out
        # Non-env argv survives intact
        assert "/tmp/x:/results" in out
        assert "image:tag" in out

    async def test_env_vars_and_mounts_forwarded_to_docker(self):
        """solver-level env_vars/mounts land in docker run -e/-v flags."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        captured_cmd: list[str] = []

        async def fake_exec(*args, **kwargs):
            captured_cmd.extend(a for a in args if isinstance(a, str))
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b""))
            proc.returncode = 0
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        env = ContainerEnvironment(image="test:latest", task="ns_mmlu")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            await env.run_batch(
                config={
                    "base_url": "http://m/v1",
                    "model": "m",
                    "api_key": "k",
                    "endpoint_type": "chat",
                    "env_vars": {"JUDGE_API_KEY": "judge-secret", "HF_TOKEN": "hf-secret"},
                    "mounts": {"/host/ruler_data": "/ruler_data", "/host/hf": "/checkpoint"},
                }
            )

        cmd_str = " ".join(captured_cmd)
        assert "JUDGE_API_KEY=judge-secret" in cmd_str
        assert "HF_TOKEN=hf-secret" in cmd_str
        assert "/host/ruler_data:/ruler_data" in cmd_str
        assert "/host/hf:/checkpoint" in cmd_str

    async def test_run_batch_container_failure(self):
        """Container exits non-zero with no results — should return empty scores, not crash."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        async def fake_exec(*args, **kwargs):
            proc = AsyncMock()
            proc.communicate = AsyncMock(return_value=(b"", b"task not found"))
            proc.returncode = 1
            proc.kill = AsyncMock()
            proc.wait = AsyncMock()
            return proc

        env = ContainerEnvironment(image="test:latest", task="bad_task")

        with patch("asyncio.create_subprocess_exec", side_effect=fake_exec):
            result = await env.run_batch(
                config={
                    "base_url": "https://api.example.com/v1",
                    "model": "test-model",
                    "endpoint_type": "chat",
                }
            )

        assert result["benchmark"]["scores"] == {}
        assert result["benchmark"]["samples"] == 0

    @pytest.mark.parametrize(
        "raw,scores,expected,case_id",
        [
            # No per-metric count → fall back to ``limit_samples`` (cap).
            (
                {
                    "config": {"params": {"limit_samples": 3, "task": "ifeval"}},
                    "results": {
                        "groups": {
                            "ifeval": {
                                "metrics": {
                                    "inst_level_loose_acc": {
                                        "scores": {
                                            "inst_level_loose_acc": {
                                                "value": 0.6,
                                                "stats": {"stderr": 0.2828},
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    },
                },
                None,  # use _extract_scores below
                3,
                "limit_samples_fallback",
            ),
            # Top-level ``samples`` reflects actual evaluated count, beats cap.
            ({"samples": 7, "config": {"params": {"limit_samples": 100}}}, {}, 7, "top_level_samples_wins"),
            # ``stats.count`` is the legacy Evaluator per-metric counter, beats cap.
            (
                {"config": {"params": {"limit_samples": 999}}},
                {"acc": {"value": 0.5, "stats": {"count": 50}}},
                50,
                "stats_count_wins",
            ),
        ],
        ids=lambda x: x if isinstance(x, str) else None,
    )
    def test_extract_sample_count_priority_chain(self, raw, scores, expected, case_id):
        """``_extract_sample_count`` prefers (in order): top-level
        ``samples``/``n_samples`` → per-metric ``stats.count`` → fallback
        ``raw.config.params.limit_samples`` (the cap)."""
        from nemo_evaluator.environments.container import ContainerEnvironment

        if scores is None:
            scores = ContainerEnvironment._extract_scores(raw)
        assert ContainerEnvironment._extract_sample_count(raw, scores) == expected

    def test_build_legacy_bundle_populates_repeats_from_results_yml(self, tmp_path):
        """v1 launcher writes back the eval-factory config under
        ``results.yml::config``.  Real production configs include
        ``config.params.extra.num_repeats`` (e.g. Ultra HMMT uses 16,
        AIME uses 64).  The bundle MUST surface this so downstream
        statistics (pass@1[avg-of-N], CIs) compute correctly — without
        it, native-env bundles get ``repeats=1`` from the schema default
        while container bundles silently drop the value."""
        import yaml as _yaml

        from nemo_evaluator.environments.container import build_legacy_bundle

        results_yml = {
            "config": {
                "type": "nemo_skills.ns_hmmt_feb2025",
                "params": {"extra": {"num_repeats": 16}},
            },
            "results": {
                "groups": {
                    "hmmt_feb25": {
                        "metrics": {
                            "pass@1": {
                                "scores": {
                                    "symbolic_correct": {"value": 100.0, "stats": {"count": 3}},
                                }
                            }
                        }
                    }
                }
            },
        }
        (tmp_path / "results.yml").write_text(_yaml.dump(results_yml))
        bundle = build_legacy_bundle("nemo-skills:dev", "nemo_skills.ns_hmmt_feb2025", tmp_path)
        assert bundle["benchmark"]["repeats"] == 16, (
            f"build_legacy_bundle must read config.params.extra.num_repeats from results.yml; "
            f"got bundle['benchmark']={bundle['benchmark']!r}"
        )

    def test_build_legacy_bundle_repeats_defaults_to_1_when_missing(self, tmp_path):
        """Harnesses without ``num_repeats`` (BBQ, terminal-bench, etc.)
        get ``repeats=1`` — matching the schema default for native-env
        bundles."""
        import yaml as _yaml

        from nemo_evaluator.environments.container import build_legacy_bundle

        results_yml = {
            "config": {"type": "bbq_small", "params": {}},
            "results": {"groups": {"bbq": {"metrics": {"acc": {"scores": {"acc": {"value": 0.5}}}}}}},
        }
        (tmp_path / "results.yml").write_text(_yaml.dump(results_yml))
        bundle = build_legacy_bundle("safety-harness:dev", "bbq_small", tmp_path)
        assert bundle["benchmark"]["repeats"] == 1


# ── SkillsEnvironment ───────────────────────────────────────────────────


class TestSkillsEnvironment:
    @patch("nemo_evaluator.environments.skills.SkillsEnvironment.__init__", return_value=None)
    def test_instantiation(self, mock_init):
        from nemo_evaluator.environments.skills import SkillsEnvironment

        SkillsEnvironment(benchmark="gsm8k")
        mock_init.assert_called_once_with(benchmark="gsm8k")


# ── HarborEnvironment ───────────────────────────────────────────────────


class TestHarborEnvironment:
    @patch("nemo_evaluator.environments.harbor.HarborEnvironment.__init__", return_value=None)
    def test_instantiation(self, mock_init):
        from nemo_evaluator.environments.harbor import HarborEnvironment

        HarborEnvironment(dataset_path="/tmp/test.json")
        mock_init.assert_called_once()


# ── Base EvalEnvironment ────────────────────────────────────────────────


class TestEvalEnvironmentBase:
    def test_dataset_property(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert env.dataset == []
        env.dataset = [{"a": 1}]
        assert len(env) == 1

    async def test_default_sandbox_specs_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.sandbox_specs() is None

    async def test_default_run_batch_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.run_batch() is None

    async def test_default_image_build_requests_none(self):
        class ConcreteEnv(EvalEnvironment):
            async def seed(self, idx):
                return SeedResult(prompt="", expected_answer="")

            async def verify(self, response, expected, sandbox=None, **meta):
                return VerifyResult(reward=0.0)

        env = ConcreteEnv()
        assert await env.image_build_requests() is None
