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
"""Tests for nemo_evaluator_launcher.common.override_sugar."""

import pytest
from omegaconf import OmegaConf

from nemo_evaluator_launcher.common.override_sugar import (
    apply_task_name_overrides,
    partition_overrides,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_cfg():
    """Config with three uniquely-named tasks."""
    return OmegaConf.create(
        {
            "evaluation": {
                "tasks": [
                    {
                        "name": "ifeval",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 1}}
                        },
                    },
                    {
                        "name": "gsm8k",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 1}}
                        },
                    },
                    {
                        "name": "mmlu",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 1}}
                        },
                    },
                ]
            }
        }
    )


@pytest.fixture
def dotted_name_cfg():
    """Config with dotted task names like 'lm-evaluation-harness.mmlu'."""
    return OmegaConf.create(
        {
            "evaluation": {
                "tasks": [
                    {
                        "name": "lm-evaluation-harness.ifeval",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 1}}
                        },
                    },
                    {
                        "name": "lm-evaluation-harness.mmlu",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 2}}
                        },
                    },
                    {
                        "name": "simple_evals.gpqa_diamond",
                        "nemo_evaluator_config": {
                            "config": {"params": {"parallelism": 1}}
                        },
                    },
                ]
            }
        }
    )


@pytest.fixture
def duplicate_suffix_cfg():
    """Config where the short suffix 'mmlu' matches two different tasks."""
    return OmegaConf.create(
        {
            "evaluation": {
                "tasks": [
                    {"name": "lm-evaluation-harness.mmlu", "nemo_evaluator_config": {"config": {"params": {"parallelism": 1}}}},
                    {"name": "nemo_skills.mmlu", "nemo_evaluator_config": {"config": {"params": {"parallelism": 1}}}},
                ]
            }
        }
    )


@pytest.fixture
def duplicate_name_cfg():
    """Config where the same task name appears more than once."""
    return OmegaConf.create(
        {
            "evaluation": {
                "tasks": [
                    {"name": "mmlu", "nemo_evaluator_config": {"config": {"params": {"parallelism": 1}}}},
                    {"name": "gsm8k", "nemo_evaluator_config": {"config": {"params": {"parallelism": 1}}}},
                    {"name": "mmlu", "nemo_evaluator_config": {"config": {"params": {"parallelism": 2}}}},
                ]
            }
        }
    )


# ---------------------------------------------------------------------------
# partition_overrides
# ---------------------------------------------------------------------------


class TestPartitionOverrides:
    def test_empty_list(self):
        hydra, sugar = partition_overrides([])
        assert hydra == []
        assert sugar == []

    def test_all_hydra(self):
        overrides = [
            "execution.output_dir=/tmp/out",
            "++evaluation.tasks.0.nemo_evaluator_config.config.params.parallelism=4",
            "target.api_endpoint.url=http://example.com",
        ]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == overrides
        assert sugar == []

    def test_all_sugar(self):
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=64",
            "++evaluation.tasks.gsm8k.nemo_evaluator_config.config.params.parallelism=32",
        ]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == []
        assert sugar == overrides

    def test_mixed(self):
        overrides = [
            "execution.output_dir=/tmp/out",
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=64",
            "++evaluation.tasks.0.nemo_evaluator_config.config.params.parallelism=4",
            "+evaluation.tasks.gsm8k.nemo_evaluator_config.config.params.extra.foo=bar",
        ]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == [
            "execution.output_dir=/tmp/out",
            "++evaluation.tasks.0.nemo_evaluator_config.config.params.parallelism=4",
        ]
        assert sugar == [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=64",
            "+evaluation.tasks.gsm8k.nemo_evaluator_config.config.params.extra.foo=bar",
        ]

    def test_index_based_pass_through(self):
        """Index-based overrides (integer after evaluation.tasks.) pass to Hydra."""
        overrides = [
            "++evaluation.tasks.1.nemo_evaluator_config.config.params.parallelism=64",
            "evaluation.tasks.0.name=new_name",
        ]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == overrides
        assert sugar == []

    def test_tilde_prefix_sugar(self):
        """~evaluation.tasks.<name>.field is sugar (delete override)."""
        overrides = ["~evaluation.tasks.mmlu.nemo_evaluator_config"]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == []
        assert sugar == overrides

    def test_non_evaluation_overrides_pass_through(self):
        overrides = [
            "deployment.type=slurm",
            "target.api_endpoint.model_id=my-model",
        ]
        hydra, sugar = partition_overrides(overrides)
        assert hydra == overrides
        assert sugar == []


# ---------------------------------------------------------------------------
# apply_task_name_overrides — basic
# ---------------------------------------------------------------------------


class TestApplyTaskNameOverrides:
    def test_simple_override(self, simple_cfg):
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=64"
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.parallelism == 64

    def test_multiple_overrides(self, simple_cfg):
        overrides = [
            "evaluation.tasks.ifeval.nemo_evaluator_config.config.params.parallelism=8",
            "evaluation.tasks.gsm8k.nemo_evaluator_config.config.params.parallelism=16",
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert cfg.evaluation.tasks[0].nemo_evaluator_config.config.params.parallelism == 8
        assert cfg.evaluation.tasks[1].nemo_evaluator_config.config.params.parallelism == 16
        # Unchanged
        assert cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.parallelism == 1

    def test_force_add_prefix(self, simple_cfg):
        overrides = [
            "++evaluation.tasks.mmlu.nemo_evaluator_config.config.params.extra.judge_url=http://judge"
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert (
            cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.extra.judge_url
            == "http://judge"
        )

    def test_no_overrides_returns_cfg(self, simple_cfg):
        cfg = apply_task_name_overrides(simple_cfg, [])
        assert cfg is simple_cfg

    def test_numeric_value(self, simple_cfg):
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=128"
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.parallelism == 128
        assert isinstance(
            cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.parallelism, int
        )

    def test_boolean_value(self, simple_cfg):
        overrides = [
            "++evaluation.tasks.mmlu.nemo_evaluator_config.config.params.extra.verbose=true"
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert (
            cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.extra.verbose is True
        )

    def test_list_value(self, simple_cfg):
        overrides = [
            "++evaluation.tasks.mmlu.nemo_evaluator_config.config.params.extra.tags=[a,b,c]"
        ]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert list(
            cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.extra.tags
        ) == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# apply_task_name_overrides — dotted task names
# ---------------------------------------------------------------------------


class TestDottedTaskNames:
    def test_full_dotted_name(self, dotted_name_cfg):
        overrides = [
            "evaluation.tasks.lm-evaluation-harness.mmlu.nemo_evaluator_config.config.params.parallelism=64"
        ]
        cfg = apply_task_name_overrides(dotted_name_cfg, overrides)
        assert cfg.evaluation.tasks[1].nemo_evaluator_config.config.params.parallelism == 64
        # Others unchanged
        assert cfg.evaluation.tasks[0].nemo_evaluator_config.config.params.parallelism == 1

    def test_suffix_match(self, dotted_name_cfg):
        """Short name 'mmlu' should match 'lm-evaluation-harness.mmlu' via suffix."""
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=32"
        ]
        cfg = apply_task_name_overrides(dotted_name_cfg, overrides)
        assert cfg.evaluation.tasks[1].nemo_evaluator_config.config.params.parallelism == 32

    def test_suffix_match_gpqa(self, dotted_name_cfg):
        overrides = [
            "evaluation.tasks.gpqa_diamond.nemo_evaluator_config.config.params.parallelism=16"
        ]
        cfg = apply_task_name_overrides(dotted_name_cfg, overrides)
        assert cfg.evaluation.tasks[2].nemo_evaluator_config.config.params.parallelism == 16


# ---------------------------------------------------------------------------
# apply_task_name_overrides — error cases
# ---------------------------------------------------------------------------


class TestErrors:
    def test_task_not_found(self, simple_cfg):
        overrides = [
            "evaluation.tasks.nonexistent.nemo_evaluator_config.config.params.parallelism=1"
        ]
        with pytest.raises(ValueError, match="not found"):
            apply_task_name_overrides(simple_cfg, overrides)

    def test_ambiguous_suffix(self, duplicate_suffix_cfg):
        """Short name 'mmlu' matches both tasks — should raise."""
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=1"
        ]
        with pytest.raises(ValueError, match="matches multiple"):
            apply_task_name_overrides(duplicate_suffix_cfg, overrides)

    def test_duplicate_task_name(self, duplicate_name_cfg):
        """Same task name at multiple indices — should raise."""
        overrides = [
            "evaluation.tasks.mmlu.nemo_evaluator_config.config.params.parallelism=1"
        ]
        with pytest.raises(ValueError, match="appears at indices"):
            apply_task_name_overrides(duplicate_name_cfg, overrides)

    def test_full_name_disambiguation(self, duplicate_suffix_cfg):
        """Using the full dotted name should work even when suffix is ambiguous."""
        overrides = [
            "evaluation.tasks.lm-evaluation-harness.mmlu.nemo_evaluator_config.config.params.parallelism=99"
        ]
        cfg = apply_task_name_overrides(duplicate_suffix_cfg, overrides)
        assert cfg.evaluation.tasks[0].nemo_evaluator_config.config.params.parallelism == 99
        # Second task unchanged
        assert cfg.evaluation.tasks[1].nemo_evaluator_config.config.params.parallelism == 1

    def test_tilde_deletes_key(self, simple_cfg):
        """~evaluation.tasks.<name>.<key> should delete the key."""
        overrides = ["~evaluation.tasks.mmlu.nemo_evaluator_config"]
        cfg = apply_task_name_overrides(simple_cfg, overrides)
        assert "nemo_evaluator_config" not in cfg.evaluation.tasks[2]
        # Other tasks untouched
        assert "nemo_evaluator_config" in cfg.evaluation.tasks[0]

    def test_tilde_nonexistent_key_raises(self, simple_cfg):
        """Deleting a key that doesn't exist should raise."""
        overrides = ["~evaluation.tasks.mmlu.no_such_key"]
        with pytest.raises(ValueError, match="does not exist"):
            apply_task_name_overrides(simple_cfg, overrides)
