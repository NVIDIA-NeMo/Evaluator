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

"""Unit tests for BYOB compiler module."""

from pathlib import Path

import pytest
import yaml

from nemo_evaluator.byob.compiler import compile_benchmark, install_benchmark


class TestCompileBenchmark:
    """Tests for compile_benchmark function."""

    def test_compile_from_file(self, tmp_path):
        """Test compiling a .py file produces correct FDF structure."""
        # Create a temp benchmark file
        benchmark_code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-compile", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(response, target, metadata):
    return {"correct": target.lower() in response.lower()}
'''
        benchmark_file = tmp_path / "test_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        # Compile the benchmark
        result = compile_benchmark(str(benchmark_file))

        # Validate structure
        assert "test_compile" in result, f"Expected 'test_compile' key, got {result.keys()}"
        fdf = result["test_compile"]

        # Framework section
        assert fdf["framework"]["name"] == "byob_test_compile", \
            f"Expected framework name 'byob_test_compile', got {fdf['framework']['name']}"
        assert fdf["framework"]["pkg_name"] == "byob_test_compile", \
            f"Expected pkg_name 'byob_test_compile', got {fdf['framework']['pkg_name']}"

        # Defaults section
        assert "command" in fdf["defaults"], "Missing 'command' in defaults"
        assert "nemo_evaluator.byob.runner" in fdf["defaults"]["command"], \
            "Command should reference nemo_evaluator.byob.runner"

        assert "config" in fdf["defaults"], "Missing 'config' in defaults"
        assert fdf["defaults"]["config"]["params"]["limit_samples"] is None
        assert fdf["defaults"]["config"]["params"]["max_new_tokens"] == 4096
        assert fdf["defaults"]["config"]["params"]["temperature"] == 0

        # Evaluations section
        assert len(fdf["evaluations"]) == 1, \
            f"Expected 1 evaluation, got {len(fdf['evaluations'])}"
        eval_entry = fdf["evaluations"][0]
        assert eval_entry["name"] == "test-compile", \
            f"Expected evaluation name 'test-compile', got {eval_entry['name']}"
        assert "byob_test_compile" in eval_entry["defaults"]["config"]["type"], \
            f"Config type should contain 'byob_test_compile', got {eval_entry['defaults']['config']['type']}"
        assert eval_entry["defaults"]["config"]["supported_endpoint_types"] == ["chat"], \
            f"Expected ['chat'], got {eval_entry['defaults']['config']['supported_endpoint_types']}"
        # Extra params are in framework-level defaults (not evaluation-level)
        # per architecture doc Section 4.1.4 (engine fallback lookup path)
        assert fdf["defaults"]["config"]["params"]["extra"]["benchmark_name"] == "test_compile", \
            "Benchmark name should be 'test_compile' in framework-level extra params"

        # Command template has correct Jinja2 placeholders
        cmd = fdf["defaults"]["command"]
        required_placeholders = [
            "config.params.extra.benchmark_module",
            "config.output_dir",
            "target.api_endpoint.url",
            "target.api_endpoint.model_id",
            "config.params.temperature",
            "config.params.max_new_tokens",
        ]
        for placeholder in required_placeholders:
            assert placeholder in cmd, \
                f"Command template missing placeholder '{placeholder}'"

    def test_compile_no_benchmarks_raises(self, tmp_path):
        """Test that module with no @benchmark decorators raises ValueError."""
        # Create a temp file with no decorators
        code = '''
def plain_function():
    return 42
'''
        benchmark_file = tmp_path / "empty_benchmark.py"
        benchmark_file.write_text(code)

        # Should raise ValueError
        with pytest.raises(ValueError, match="No benchmarks found"):
            compile_benchmark(str(benchmark_file))

    def test_compile_multi_benchmark(self, tmp_path):
        """Test module with 2 benchmarks produces 2 FDF entries."""
        code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="bench-one", dataset="d1.jsonl", prompt="{q}")
@scorer
def scorer_one(response, target, metadata):
    return {"correct": True}

@benchmark(name="bench-two", dataset="d2.jsonl", prompt="{q}")
@scorer
def scorer_two(response, target, metadata):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "multi_benchmark.py"
        benchmark_file.write_text(code)

        result = compile_benchmark(str(benchmark_file))

        # Should have 2 entries
        assert len(result) == 2, f"Expected 2 benchmarks, got {len(result)}"
        assert "bench_one" in result, "Missing 'bench_one' in compiled result"
        assert "bench_two" in result, "Missing 'bench_two' in compiled result"
        assert result["bench_one"]["evaluations"][0]["name"] == "bench-one", \
            f"Expected name 'bench-one', got {result['bench_one']['evaluations'][0]['name']}"
        assert result["bench_two"]["evaluations"][0]["name"] == "bench-two", \
            f"Expected name 'bench-two', got {result['bench_two']['evaluations'][0]['name']}"


class TestInstallBenchmark:
    """Tests for install_benchmark function."""

    def test_install_creates_package(self, tmp_path):
        """Test that install_benchmark creates correct directory tree with all required files."""
        # Create a minimal FDF dict
        fdf = {
            "framework": {"name": "byob_test_pkg", "pkg_name": "byob_test_pkg"},
            "defaults": {
                "command": "python -m nemo_evaluator.byob.runner ...",
                "config": {
                    "params": {
                        "limit_samples": None,
                        "max_new_tokens": 4096,
                        "temperature": 0,
                        "extra": {
                            "benchmark_module": "/path/to/module.py",
                            "benchmark_name": "test_pkg",
                            "dataset": "/path/to/data.jsonl",
                        },
                    }
                },
                "target": {"api_endpoint": {}},
            },
            "evaluations": [{
                "name": "test-pkg",
                "description": "BYOB benchmark: test-pkg",
                "defaults": {
                    "config": {
                        "type": "byob_test_pkg.test-pkg",
                        "supported_endpoint_types": ["chat"],
                        "params": {
                            "extra": {
                                "benchmark_module": "/path/to/module.py",
                                "benchmark_name": "test_pkg",
                                "dataset": "/path/to/data.jsonl",
                            }
                        }
                    }
                }
            }]
        }

        # Install benchmark
        pkg_dir = install_benchmark("test_pkg", fdf, install_dir=str(tmp_path))
        pkg_path = Path(pkg_dir)

        # Validate directory structure
        assert (pkg_path / "pyproject.toml").is_file(), \
            "Missing pyproject.toml"
        assert (pkg_path / "core_evals" / "__init__.py").is_file(), \
            "Missing core_evals/__init__.py"
        assert (pkg_path / "core_evals" / "byob_test_pkg" / "__init__.py").is_file(), \
            "Missing core_evals/byob_test_pkg/__init__.py"
        assert (pkg_path / "core_evals" / "byob_test_pkg" / "framework.yml").is_file(), \
            "Missing framework.yml"
        assert (pkg_path / "core_evals" / "byob_test_pkg" / "output.py").is_file(), \
            "Missing output.py"

        # Validate framework.yml is valid YAML
        with open(pkg_path / "core_evals" / "byob_test_pkg" / "framework.yml") as f:
            fw = yaml.safe_load(f)
        assert fw is not None, "framework.yml failed to parse"
        assert "framework" in fw, "framework.yml missing 'framework' key"

        # Validate core_evals/__init__.py contains pkgutil.extend_path
        init_content = (pkg_path / "core_evals" / "__init__.py").read_text()
        assert "extend_path" in init_content, \
            "core_evals/__init__.py should use pkgutil.extend_path for namespace"

        # Validate output.py contains parse_output function
        output_content = (pkg_path / "core_evals" / "byob_test_pkg" / "output.py").read_text()
        assert "def parse_output" in output_content, \
            "output.py should define parse_output function"
        assert "byob_results.json" in output_content, \
            "output.py should reference byob_results.json"

        # Validate pyproject.toml contains correct package name
        toml_content = (pkg_path / "pyproject.toml").read_text()
        assert "core-evals-byob_test_pkg" in toml_content, \
            "pyproject.toml should contain package name core-evals-byob_test_pkg"
