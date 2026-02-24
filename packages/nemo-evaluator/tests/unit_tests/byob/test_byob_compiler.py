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

import sys
from pathlib import Path

import pytest
import yaml

from nemo_evaluator.contrib.byob.compiler import compile_benchmark, install_benchmark
from nemo_evaluator.contrib.byob.defaults import DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE


class TestCompileBenchmark:
    """Tests for compile_benchmark function."""

    def test_compile_from_file(self, tmp_path):
        """Test compiling a .py file produces correct FDF structure."""
        # Create a temp benchmark file
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="test-compile", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(sample):
    return {"correct": sample.target.lower() in sample.response.lower()}
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
        assert "nemo_evaluator.contrib.byob.runner" in fdf["defaults"]["command"], \
            "Command should reference nemo_evaluator.contrib.byob.runner"

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
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="bench-one", dataset="d1.jsonl", prompt="{q}")
@scorer
def scorer_one(sample):
    return {"correct": True}

@benchmark(name="bench-two", dataset="d2.jsonl", prompt="{q}")
@scorer
def scorer_two(sample):
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
                "command": "python -m nemo_evaluator.contrib.byob.runner ...",
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


class TestCompileWithRequirements:
    """Tests for requirements field in compile_benchmark output."""

    def test_compile_benchmark_with_inline_requirements(self, tmp_path):
        """Test that inline requirements list is propagated to FDF extra params."""
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(
    name="test-reqs-inline",
    dataset="data.jsonl",
    prompt="Q: {q}\\nA:",
    requirements=["numpy>=1.0", "pandas"],
)
@scorer
def check(sample):
    return {"correct": sample.target.lower() in sample.response.lower()}
'''
        benchmark_file = tmp_path / "reqs_inline_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        result = compile_benchmark(str(benchmark_file))

        assert "test_reqs_inline" in result, \
            f"Expected 'test_reqs_inline' key, got {result.keys()}"
        fdf = result["test_reqs_inline"]

        extra = fdf["defaults"]["config"]["params"]["extra"]
        assert "requirements" in extra, \
            "FDF extra params should contain 'requirements' key"
        assert extra["requirements"] == ["numpy>=1.0", "pandas"], \
            f"Expected ['numpy>=1.0', 'pandas'], got {extra['requirements']}"

    def test_compile_benchmark_with_no_requirements(self, tmp_path):
        """Test that omitting requirements produces an empty list in FDF extra params."""
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="test-no-reqs", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(sample):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "no_reqs_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        result = compile_benchmark(str(benchmark_file))

        assert "test_no_reqs" in result, \
            f"Expected 'test_no_reqs' key, got {result.keys()}"
        fdf = result["test_no_reqs"]

        extra = fdf["defaults"]["config"]["params"]["extra"]
        assert "requirements" in extra, \
            "FDF extra params should contain 'requirements' key even when empty"
        assert extra["requirements"] == [], \
            f"Expected empty list, got {extra['requirements']}"

    def test_compile_benchmark_with_requirements_file(self, tmp_path):
        """Test that a requirements.txt file path is resolved and parsed into FDF."""
        # Create a requirements.txt file
        reqs_file = tmp_path / "requirements.txt"
        reqs_file.write_text("# Comment line\nnumpy>=1.0\npandas\n\nscikit-learn>=1.2\n")

        benchmark_code = f'''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(
    name="test-reqs-file",
    dataset="data.jsonl",
    prompt="Q: {{q}}\\nA:",
    requirements="{reqs_file}",
)
@scorer
def check(sample):
    return {{"correct": True}}
'''
        benchmark_file = tmp_path / "reqs_file_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        result = compile_benchmark(str(benchmark_file))

        assert "test_reqs_file" in result, \
            f"Expected 'test_reqs_file' key, got {result.keys()}"
        fdf = result["test_reqs_file"]

        extra = fdf["defaults"]["config"]["params"]["extra"]
        assert "requirements" in extra, \
            "FDF extra params should contain 'requirements' key"
        assert extra["requirements"] == ["numpy>=1.0", "pandas", "scikit-learn>=1.2"], \
            f"Expected parsed requirements list, got {extra['requirements']}"


class TestInstallWithRequirements:
    """Tests for requirements propagation through install_benchmark to pyproject.toml."""

    def _make_fdf(self, requirements=None):
        """Helper to build a minimal FDF dict with optional requirements in extra params."""
        extra = {
            "benchmark_module": "/path/to/module.py",
            "benchmark_name": "test_pkg",
            "dataset": "/path/to/data.jsonl",
        }
        if requirements is not None:
            extra["requirements"] = requirements

        return {
            "framework": {"name": "byob_test_pkg", "pkg_name": "byob_test_pkg"},
            "defaults": {
                "command": "python -m nemo_evaluator.contrib.byob.runner ...",
                "config": {
                    "params": {
                        "limit_samples": None,
                        "max_new_tokens": 4096,
                        "temperature": 0,
                        "extra": extra,
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
                    }
                }
            }]
        }

    def test_install_with_user_requirements(self, tmp_path):
        """Test that user requirements appear in pyproject.toml dependencies."""
        fdf = self._make_fdf(requirements=["numpy>=1.0"])
        pkg_dir = install_benchmark("test_pkg", fdf, install_dir=str(tmp_path))

        toml_content = (Path(pkg_dir) / "pyproject.toml").read_text()

        assert '"nemo-evaluator"' in toml_content, \
            "pyproject.toml should always include nemo-evaluator dependency"
        assert '"numpy>=1.0"' in toml_content, \
            f"pyproject.toml should contain numpy>=1.0 dependency, got:\n{toml_content}"

        # Verify the full dependencies line has both deps
        assert 'dependencies = ["nemo-evaluator", "numpy>=1.0"]' in toml_content, \
            f"Expected dependencies with both nemo-evaluator and numpy>=1.0, got:\n{toml_content}"

    def test_install_without_user_requirements(self, tmp_path):
        """Test that FDF without requirements key produces pyproject.toml with only nemo-evaluator."""
        fdf = self._make_fdf(requirements=None)
        # Remove requirements key entirely from extra to simulate old FDFs
        fdf["defaults"]["config"]["params"]["extra"].pop("requirements", None)

        pkg_dir = install_benchmark("test_pkg", fdf, install_dir=str(tmp_path))

        toml_content = (Path(pkg_dir) / "pyproject.toml").read_text()

        assert 'dependencies = ["nemo-evaluator"]' in toml_content, \
            f"Expected only nemo-evaluator dependency, got:\n{toml_content}"

    def test_install_with_empty_requirements(self, tmp_path):
        """Test that empty requirements list produces pyproject.toml with only nemo-evaluator."""
        fdf = self._make_fdf(requirements=[])
        pkg_dir = install_benchmark("test_pkg", fdf, install_dir=str(tmp_path))

        toml_content = (Path(pkg_dir) / "pyproject.toml").read_text()

        assert 'dependencies = ["nemo-evaluator"]' in toml_content, \
            f"Expected only nemo-evaluator dependency when requirements is empty, got:\n{toml_content}"


class TestCompilerSysPathCleanup:
    """Tests for sys.path and sys.modules cleanup after compile_benchmark."""

    def test_sys_path_restored_after_compile(self, tmp_path):
        """Validate sys.path is restored to its original state after compilation."""
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="path-test", dataset="data.jsonl", prompt="{q}")
@scorer
def check(sample):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "path_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        saved_path = sys.path[:]

        compile_benchmark(str(benchmark_file))

        assert sys.path == saved_path, (
            f"sys.path was not restored after compile_benchmark. "
            f"Expected {len(saved_path)} entries, got {len(sys.path)}"
        )

    def test_sys_modules_cleaned_after_compile(self, tmp_path):
        """Validate user benchmark module is removed from sys.modules after compilation."""
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="modules-test", dataset="data.jsonl", prompt="{q}")
@scorer
def check(sample):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "modules_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        compile_benchmark(str(benchmark_file))

        # The user module should have been cleaned up from sys.modules
        assert "modules_benchmark" not in sys.modules, (
            "User benchmark module 'modules_benchmark' should have been removed "
            "from sys.modules after compile_benchmark"
        )

    def test_cleanup_on_compile_error(self, tmp_path):
        """Validate sys.path is restored even when compile_benchmark raises."""
        bad_code = '''
raise RuntimeError("intentional error during import")
'''
        benchmark_file = tmp_path / "bad_benchmark.py"
        benchmark_file.write_text(bad_code)

        saved_path = sys.path[:]

        with pytest.raises(RuntimeError, match="intentional error during import"):
            compile_benchmark(str(benchmark_file))

        assert sys.path == saved_path, (
            f"sys.path was not restored after failed compile_benchmark. "
            f"Expected {len(saved_path)} entries, got {len(sys.path)}"
        )


class TestBuildFdfHelper:
    """Tests for the _build_fdf helper function and FDF structural correctness."""

    def test_fdf_uses_defaults_constants(self, tmp_path):
        """Validate max_new_tokens and temperature in FDF match defaults.py constants."""
        benchmark_code = '''
from nemo_evaluator.contrib.byob import benchmark, scorer

@benchmark(name="defaults-fdf", dataset="data.jsonl", prompt="{q}")
@scorer
def check(sample):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "defaults_fdf_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        result = compile_benchmark(str(benchmark_file))
        fdf = result["defaults_fdf"]

        params = fdf["defaults"]["config"]["params"]

        assert params["max_new_tokens"] == DEFAULT_MAX_TOKENS, (
            f"FDF max_new_tokens should be {DEFAULT_MAX_TOKENS}, "
            f"got {params['max_new_tokens']}"
        )
        assert params["temperature"] == DEFAULT_TEMPERATURE, (
            f"FDF temperature should be {DEFAULT_TEMPERATURE}, "
            f"got {params['temperature']}"
        )
