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


class TestCompileNativeMode:
    """Tests for native mode compilation extensions."""

    def test_compiler_native_mode_fdf(self, tmp_path):
        """Test compiling with execution_mode='native' produces correct FDF structure."""
        # Create a temp benchmark file
        benchmark_code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-native", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(response, target, metadata):
    return {"correct": target.lower() in response.lower()}
'''
        benchmark_file = tmp_path / "native_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        # Compile with native mode
        result = compile_benchmark(str(benchmark_file), execution_mode="native")

        # Validate structure
        assert "test_native" in result, f"Expected 'test_native' key, got {result.keys()}"
        fdf = result["test_native"]

        # Verify execution_mode is set to native
        assert fdf["defaults"].get("execution_mode") == "native", \
            f"FDF should contain execution_mode='native', got {fdf['defaults'].get('execution_mode')!r}"

        # Verify command is NOT present in native mode
        assert "command" not in fdf["defaults"], \
            "Native mode FDF should not contain 'command' field"

        # Framework section should still be correct
        assert fdf["framework"]["name"] == "byob_test_native", \
            f"Expected framework name 'byob_test_native', got {fdf['framework']['name']}"

        # Config params should still be present
        assert fdf["defaults"]["config"]["params"]["limit_samples"] is None
        assert fdf["defaults"]["config"]["params"]["max_new_tokens"] == 4096
        assert fdf["defaults"]["config"]["params"]["temperature"] == 0

        # Extra params should contain benchmark info
        assert fdf["defaults"]["config"]["params"]["extra"]["benchmark_name"] == "test_native", \
            "Benchmark name should be 'test_native' in extra params"

    def test_compiler_default_subprocess_mode_fdf(self, tmp_path):
        """Test default compilation (no execution_mode arg) produces subprocess mode FDF."""
        benchmark_code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-default", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(response, target, metadata):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "default_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        # Compile WITHOUT execution_mode argument (should default to subprocess)
        result = compile_benchmark(str(benchmark_file))

        fdf = result["test_default"]

        # Should have command field (subprocess mode)
        assert "command" in fdf["defaults"], \
            "Default (subprocess) mode should include 'command' field"
        assert "nemo_evaluator.byob.runner" in fdf["defaults"]["command"], \
            "Default mode command should reference subprocess runner"

        # Should NOT have execution_mode field, or it should be absent
        # (backward compatibility - existing FDFs don't have execution_mode)
        mode = fdf["defaults"].get("execution_mode")
        assert mode is None or mode == "subprocess", \
            f"Default compilation should not set execution_mode='native', got {mode!r}"

    def test_compiler_explicit_subprocess_mode(self, tmp_path):
        """Test explicit execution_mode='subprocess' produces subprocess FDF."""
        benchmark_code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-subprocess", dataset="data.jsonl", prompt="{q}")
@scorer
def check(response, target, metadata):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "subprocess_benchmark.py"
        benchmark_file.write_text(benchmark_code)

        result = compile_benchmark(str(benchmark_file), execution_mode="subprocess")

        fdf = result["test_subprocess"]

        # Should have command field
        assert "command" in fdf["defaults"], \
            "Subprocess mode should include 'command' field"

    def test_install_native_mode_framework_yml(self, tmp_path):
        """Test installing native mode FDF writes execution_mode='native' to framework.yml."""
        # Create a native mode FDF
        fdf_native = {
            "framework": {"name": "byob_native_pkg", "pkg_name": "byob_native_pkg"},
            "defaults": {
                "execution_mode": "native",
                # No "command" field for native mode
                "config": {
                    "params": {
                        "limit_samples": None,
                        "max_new_tokens": 4096,
                        "temperature": 0,
                        "extra": {
                            "benchmark_module": "/path/to/module.py",
                            "benchmark_name": "native_pkg",
                            "dataset": "/path/to/data.jsonl",
                        },
                    }
                },
                "target": {"api_endpoint": {}},
            },
            "evaluations": [{
                "name": "native-pkg",
                "description": "BYOB benchmark: native-pkg",
                "defaults": {
                    "config": {
                        "type": "byob_native_pkg.native-pkg",
                        "supported_endpoint_types": ["chat"],
                    }
                }
            }]
        }

        # Install benchmark
        pkg_dir = install_benchmark("native_pkg", fdf_native, install_dir=str(tmp_path))

        # Validate framework.yml contains execution_mode
        fw_path = Path(pkg_dir) / "core_evals" / "byob_native_pkg" / "framework.yml"
        assert fw_path.is_file(), f"framework.yml should exist at {fw_path}"

        with open(fw_path) as f:
            fw = yaml.safe_load(f)

        # Verify execution_mode is preserved
        assert fw["defaults"].get("execution_mode") == "native", \
            f"framework.yml should contain execution_mode='native', got {fw['defaults'].get('execution_mode')!r}"

        # Verify command field is absent
        assert "command" not in fw["defaults"], \
            "Native mode framework.yml should not contain 'command' field"

    def test_install_native_mode_no_output_py(self, tmp_path):
        """Test native mode installation does NOT create output.py (not needed for native)."""
        fdf_native = {
            "framework": {"name": "byob_native_pkg2", "pkg_name": "byob_native_pkg2"},
            "defaults": {
                "execution_mode": "native",
                "config": {
                    "params": {
                        "limit_samples": None,
                        "max_new_tokens": 4096,
                        "temperature": 0,
                        "extra": {
                            "benchmark_module": "/path/to/module.py",
                            "benchmark_name": "native_pkg2",
                            "dataset": "/path/to/data.jsonl",
                        },
                    }
                },
                "target": {"api_endpoint": {}},
            },
            "evaluations": [{
                "name": "native-pkg2",
                "description": "BYOB benchmark: native-pkg2",
                "defaults": {
                    "config": {
                        "type": "byob_native_pkg2.native-pkg2",
                        "supported_endpoint_types": ["chat"],
                    }
                }
            }]
        }

        pkg_dir = install_benchmark("native_pkg2", fdf_native, install_dir=str(tmp_path))

        # output.py should NOT exist for native mode
        output_py_path = Path(pkg_dir) / "core_evals" / "byob_native_pkg2" / "output.py"
        assert not output_py_path.exists(), \
            "Native mode should NOT generate output.py (returns EvaluationResult directly)"

    def test_install_subprocess_mode_has_output_py(self, tmp_path):
        """Test subprocess mode installation DOES create output.py (needed for parsing)."""
        fdf_subprocess = {
            "framework": {"name": "byob_subprocess_pkg", "pkg_name": "byob_subprocess_pkg"},
            "defaults": {
                # No execution_mode = defaults to subprocess
                "command": "python -m nemo_evaluator.byob.runner ...",
                "config": {
                    "params": {
                        "limit_samples": None,
                        "max_new_tokens": 4096,
                        "temperature": 0,
                        "extra": {
                            "benchmark_module": "/path/to/module.py",
                            "benchmark_name": "subprocess_pkg",
                            "dataset": "/path/to/data.jsonl",
                        },
                    }
                },
                "target": {"api_endpoint": {}},
            },
            "evaluations": [{
                "name": "subprocess-pkg",
                "description": "BYOB benchmark: subprocess-pkg",
                "defaults": {
                    "config": {
                        "type": "byob_subprocess_pkg.subprocess-pkg",
                        "supported_endpoint_types": ["chat"],
                    }
                }
            }]
        }

        pkg_dir = install_benchmark("subprocess_pkg", fdf_subprocess, install_dir=str(tmp_path))

        # output.py SHOULD exist for subprocess mode
        output_py_path = Path(pkg_dir) / "core_evals" / "byob_subprocess_pkg" / "output.py"
        assert output_py_path.is_file(), \
            "Subprocess mode MUST generate output.py for parsing byob_results.json"

    def test_compiler_native_mode_multi_benchmark(self, tmp_path):
        """Test native mode compilation with multiple benchmarks in one module."""
        code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="bench-one-native", dataset="d1.jsonl", prompt="{q}")
@scorer
def scorer_one(response, target, metadata):
    return {"correct": True}

@benchmark(name="bench-two-native", dataset="d2.jsonl", prompt="{q}")
@scorer
def scorer_two(response, target, metadata):
    return {"correct": True}
'''
        benchmark_file = tmp_path / "multi_native_benchmark.py"
        benchmark_file.write_text(code)

        result = compile_benchmark(str(benchmark_file), execution_mode="native")

        # Should have 2 entries, both with native mode
        assert len(result) == 2, f"Expected 2 benchmarks, got {len(result)}"
        assert "bench_one_native" in result
        assert "bench_two_native" in result

        # Both should have execution_mode='native'
        assert result["bench_one_native"]["defaults"].get("execution_mode") == "native"
        assert result["bench_two_native"]["defaults"].get("execution_mode") == "native"

        # Neither should have command field
        assert "command" not in result["bench_one_native"]["defaults"]
        assert "command" not in result["bench_two_native"]["defaults"]


class TestCompileWithRequirements:
    """Tests for requirements field in compile_benchmark output."""

    def test_compile_benchmark_with_inline_requirements(self, tmp_path):
        """Test that inline requirements list is propagated to FDF extra params."""
        benchmark_code = '''
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="test-reqs-inline",
    dataset="data.jsonl",
    prompt="Q: {q}\\nA:",
    requirements=["numpy>=1.0", "pandas"],
)
@scorer
def check(response, target, metadata):
    return {"correct": target.lower() in response.lower()}
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
from nemo_evaluator.byob import benchmark, scorer

@benchmark(name="test-no-reqs", dataset="data.jsonl", prompt="Q: {q}\\nA:")
@scorer
def check(response, target, metadata):
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
from nemo_evaluator.byob import benchmark, scorer

@benchmark(
    name="test-reqs-file",
    dataset="data.jsonl",
    prompt="Q: {{q}}\\nA:",
    requirements="{reqs_file}",
)
@scorer
def check(response, target, metadata):
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
                "command": "python -m nemo_evaluator.byob.runner ...",
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
