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
import pytest

from nemo_evaluator.config.gate_policy import (
    BenchmarkGateDefaults,
    BenchmarkGateEntry,
    Direction,
    GatePolicy,
    Tier,
    default_policy,
    load_gate_policy,
)


class TestGatePolicyParsing:
    def test_minimal_policy(self):
        p = GatePolicy.model_validate({"version": 1})
        assert p.version == 1
        assert p.defaults.tier == Tier.supporting
        assert p.defaults.max_drop == 0.015
        assert p.benchmarks == {}

    def test_full_policy(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {
                    "tier": "critical",
                    "max_drop": 0.01,
                    "max_relative_drop": 0.02,
                    "relative_guard_below": 0.2,
                    "metric": "mean_reward",
                    "direction": "higher_is_better",
                },
                "benchmarks": {
                    "mmlu": {"tier": "supporting", "max_drop": 0.015},
                    "gpqa": {},
                },
            }
        )
        assert p.defaults.tier == Tier.critical
        assert p.defaults.max_relative_drop == 0.02
        assert "mmlu" in p.benchmarks
        assert p.benchmarks["mmlu"].tier == Tier.supporting

    def test_invalid_version(self):
        with pytest.raises(ValueError, match="Unsupported gate policy version"):
            GatePolicy.model_validate({"version": 2})

    def test_negative_max_drop(self):
        with pytest.raises(ValueError, match="max_drop must be >= 0"):
            BenchmarkGateDefaults(max_drop=-0.01)

    def test_negative_max_relative_drop(self):
        with pytest.raises(ValueError, match="max_relative_drop must be >= 0"):
            BenchmarkGateDefaults(max_relative_drop=-0.5)

    def test_relative_guard_below_out_of_range(self):
        with pytest.raises(ValueError, match="relative_guard_below must be in"):
            BenchmarkGateDefaults(relative_guard_below=1.5)

    def test_extra_field_forbidden(self):
        with pytest.raises(ValueError):
            GatePolicy.model_validate({"version": 1, "unknown_field": True})

    def test_extra_field_in_entry(self):
        with pytest.raises(ValueError):
            BenchmarkGateEntry.model_validate({"tier": "critical", "bogus": 42})

    def test_three_tiers(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "benchmarks": {
                    "a": {"tier": "critical"},
                    "b": {"tier": "supporting"},
                    "c": {"tier": "advisory"},
                },
            }
        )
        assert p.benchmarks["a"].tier == Tier.critical
        assert p.benchmarks["b"].tier == Tier.supporting
        assert p.benchmarks["c"].tier == Tier.advisory

    def test_direction_parses_both_values(self):
        d1 = BenchmarkGateDefaults(direction="higher_is_better")
        d2 = BenchmarkGateDefaults(direction="lower_is_better")
        assert d1.direction == Direction.higher_is_better
        assert d2.direction == Direction.lower_is_better


class TestResolve:
    def test_resolve_with_overrides(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"tier": "supporting", "max_drop": 0.015, "metric": "mean_reward"},
                "benchmarks": {"mmlu": {"tier": "critical", "max_drop": 0.01}},
            }
        )
        resolved = p.resolve("mmlu")
        assert resolved.tier == Tier.critical
        assert resolved.max_drop == 0.01
        assert resolved.metric == "mean_reward"  # inherited

    def test_resolve_unknown_benchmark(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"tier": "supporting", "max_drop": 0.015},
            }
        )
        resolved = p.resolve("unknown_bench")
        assert resolved.tier == Tier.supporting
        assert resolved.max_drop == 0.015

    def test_resolve_empty_entry_inherits_all(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"max_drop": 0.02, "direction": "lower_is_better"},
                "benchmarks": {"ppl": {}},
            }
        )
        resolved = p.resolve("ppl")
        assert resolved.max_drop == 0.02
        assert resolved.direction == Direction.lower_is_better


class TestRequiredBenchmarks:
    def test_excludes_advisory(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "benchmarks": {
                    "mmlu": {"tier": "critical"},
                    "gpqa": {"tier": "supporting"},
                    "ifeval": {"tier": "advisory"},
                },
            }
        )
        required = p.required_benchmarks()
        assert required == {"mmlu", "gpqa"}

    def test_inherits_default_tier(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"tier": "critical"},
                "benchmarks": {"a": {}, "b": {"tier": "advisory"}},
            }
        )
        assert p.required_benchmarks() == {"a"}

    def test_empty_benchmarks(self):
        p = default_policy()
        assert p.required_benchmarks() == set()


class TestGateValidation:
    def test_validate_for_gate_accepts_supported_required_metrics(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"metric": "mean_reward"},
                "benchmarks": {"mmlu": {"tier": "critical"}},
            }
        )
        p.validate_for_gate({"mean_reward", "pass@1"})

    def test_validate_for_gate_rejects_missing_metric(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "benchmarks": {"mmlu": {"tier": "critical"}},
            }
        )
        with pytest.raises(ValueError, match="explicit metric"):
            p.validate_for_gate({"mean_reward", "pass@1"})

    def test_validate_for_gate_rejects_unsupported_metric(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "defaults": {"metric": "scorer:rouge"},
                "benchmarks": {"mmlu": {"tier": "critical"}},
            }
        )
        with pytest.raises(ValueError, match="unsupported metric"):
            p.validate_for_gate({"mean_reward", "pass@1"})

    def test_validate_for_gate_ignores_advisory_without_metric(self):
        p = GatePolicy.model_validate(
            {
                "version": 1,
                "benchmarks": {"triviaqa": {"tier": "advisory"}},
            }
        )
        p.validate_for_gate({"mean_reward", "pass@1"})


class TestLoadFromFile:
    def test_round_trip(self, tmp_path):
        policy_yaml = tmp_path / "policy.yaml"
        policy_yaml.write_text(
            "version: 1\n"
            "defaults:\n"
            "  tier: critical\n"
            "  max_drop: 0.01\n"
            "benchmarks:\n"
            "  mmlu:\n"
            "    tier: supporting\n"
            "    max_drop: 0.015\n"
        )
        p = load_gate_policy(policy_yaml)
        assert p.defaults.tier == Tier.critical
        assert p.benchmarks["mmlu"].max_drop == 0.015

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_gate_policy("/nonexistent/path.yaml")

    def test_invalid_yaml_content(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("just a string")
        with pytest.raises(ValueError, match="YAML mapping"):
            load_gate_policy(f)
