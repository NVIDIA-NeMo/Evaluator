# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Tests for nemo-skills plugin discovery and configuration."""

from pathlib import Path

import pytest
import yaml


class TestPluginDiscovery:
    """Tests for namespace package discovery (T-064, T-065, T-107, T-108)."""

    def test_t064_namespace_package_discovery(self):
        """T-064: core_evals.nemo_skills namespace package is discoverable (AC-032, INV-011, R-035)."""
        # The package is installed as part of nemo-evaluator in editable mode.
        # Verify that the core_evals/nemo_skills directory exists in the source tree.
        src_root = Path(__file__).parent.parent.parent.parent / "src"
        ns_pkg_path = src_root / "core_evals" / "nemo_skills"
        assert ns_pkg_path.exists(), f"Namespace package not found at {ns_pkg_path}"
        assert (ns_pkg_path / "__init__.py").exists()
        assert (ns_pkg_path / "framework.yml").exists()

    def test_t065_framework_evaluations_parsed(self):
        """T-065: framework.yml parses without error and contains expected structure (AC-032)."""
        src_root = Path(__file__).parent.parent.parent.parent / "src"
        framework_path = src_root / "core_evals" / "nemo_skills" / "framework.yml"
        assert framework_path.exists()
        with open(framework_path) as f:
            config = yaml.safe_load(f)
        assert "framework" in config
        assert "evaluations" in config
        assert len(config["evaluations"]) > 0

    def test_t107_framework_yml_native_entrypoint_specified(self):
        """T-107: All native-mode evaluations specify native_entrypoint."""
        src_root = Path(__file__).parent.parent.parent.parent / "src"
        framework_path = src_root / "core_evals" / "nemo_skills" / "framework.yml"
        with open(framework_path) as f:
            config = yaml.safe_load(f)
        defaults = config.get("defaults", {})
        for eval_entry in config["evaluations"]:
            exec_mode = eval_entry.get("execution_mode", defaults.get("execution_mode"))
            if exec_mode == "native":
                entrypoint = eval_entry.get(
                    "native_entrypoint",
                    defaults.get("native_entrypoint"),
                )
                assert entrypoint is not None, (
                    f"Evaluation '{eval_entry.get('name')}' is native mode but has no native_entrypoint"
                )

    def test_t108_framework_yml_isolation_required_subprocess_only(self):
        """T-108: All entries with isolation_required=true have execution_mode=subprocess."""
        src_root = Path(__file__).parent.parent.parent.parent / "src"
        framework_path = src_root / "core_evals" / "nemo_skills" / "framework.yml"
        with open(framework_path) as f:
            config = yaml.safe_load(f)
        defaults = config.get("defaults", {})
        for eval_entry in config["evaluations"]:
            isolation = eval_entry.get("isolation_required", False)
            if isolation:
                exec_mode = eval_entry.get("execution_mode", defaults.get("execution_mode"))
                assert exec_mode == "subprocess", (
                    f"Evaluation '{eval_entry.get('name')}' requires isolation but is not subprocess mode"
                )
