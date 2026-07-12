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
"""DockerExecutor must write the full_config.yaml snapshot to the output dir."""

from nemo_evaluator.config import EvalConfig
from nemo_evaluator.executors.docker_executor import DockerExecutor


def _raw_config(out_dir):
    return {
        "services": {
            "m": {
                "type": "api",
                "url": "https://example.com/v1/chat/completions",
                "protocol": "chat_completions",
                "model": "test-model",
                "api_key": "${TEST_API_KEY}",
            }
        },
        "benchmarks": [{"name": "gsm8k", "max_problems": 1, "solver": {"type": "simple", "service": "m"}}],
        "cluster": {"type": "docker"},
        "output": {"dir": str(out_dir)},
    }


def test_dry_run_writes_config_snapshot(tmp_path, monkeypatch):
    monkeypatch.delenv("NEL_INNER_EXECUTION", raising=False)
    out_dir = tmp_path / "out"
    raw = _raw_config(out_dir)
    config = EvalConfig.model_validate(raw)
    # The CLI loader attaches the pre-expansion composed dict; simulate it.
    config._composed_raw = raw

    # dry_run writes .docker.env, _docker_config.json and the snapshot,
    # then stops before `docker run`.
    DockerExecutor().run(config, dry_run=True)

    text = (out_dir / "full_config.yaml").read_text()
    assert "nemo-evaluator version" in text  # provenance header present
    assert "${TEST_API_KEY}" in text  # env ref survives unexpanded


def test_dry_run_no_snapshot_without_composed_raw(tmp_path, monkeypatch):
    """Programmatic configs hold resolved secrets — the executor writes no snapshot."""
    monkeypatch.delenv("NEL_INNER_EXECUTION", raising=False)
    out_dir = tmp_path / "out"
    config = EvalConfig.model_validate(_raw_config(out_dir))

    DockerExecutor().run(config, dry_run=True)

    assert not (out_dir / "full_config.yaml").exists()
