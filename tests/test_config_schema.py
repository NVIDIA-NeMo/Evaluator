"""Tests for eval config schema validation and env var expansion."""
import pytest

from nemo_evaluator.eval.config import (
    EvalConfig,
    ModelConfig,
    BenchmarkConfig,
    ServiceConfig,
    ClusterConfig,
    OutputConfig,
    parse_eval_config,
    _expand_env,
)


class TestEnvExpansion:
    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("TEST_URL", "http://example.com")
        assert _expand_env("${TEST_URL}") == "http://example.com"

    def test_default_value(self):
        result = _expand_env("${NONEXISTENT_VAR:-fallback}")
        assert result == "fallback"

    def test_nested_dict(self, monkeypatch):
        monkeypatch.setenv("MY_MODEL", "gpt-4")
        result = _expand_env({"model": "${MY_MODEL}", "url": "${MISSING:-default}"})
        assert result == {"model": "gpt-4", "url": "default"}

    def test_list_expansion(self, monkeypatch):
        monkeypatch.setenv("BENCH", "gsm8k")
        result = _expand_env(["${BENCH}", "triviaqa"])
        assert result == ["gsm8k", "triviaqa"]

    def test_non_string_passthrough(self):
        assert _expand_env(42) == 42
        assert _expand_env(3.14) == 3.14
        assert _expand_env(None) is None


class TestParseEvalConfig:
    def test_simple_mode(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "benchmarks": [{"name": "gsm8k"}],
        }
        cfg = parse_eval_config(raw)
        assert cfg.is_simple
        assert cfg.model.url == "http://localhost:8000/v1"
        assert len(cfg.benchmarks) == 1
        assert cfg.benchmarks[0].name == "gsm8k"

    def test_simple_mode_with_options(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "benchmarks": [
                {"name": "gsm8k", "repeats": 5, "max_problems": 100},
                {"name": "lm-eval://aime2025", "repeats": 20},
            ],
        }
        cfg = parse_eval_config(raw)
        assert len(cfg.benchmarks) == 2
        assert cfg.benchmarks[0].repeats == 5
        assert cfg.benchmarks[0].max_problems == 100
        assert cfg.benchmarks[1].name == "lm-eval://aime2025"

    def test_advanced_mode(self):
        raw = {
            "services": {
                "evaluated": {"type": "vllm", "model": "Qwen/Qwen3.5-9B", "port": 8000},
            },
            "benchmarks": [{"name": "gsm8k", "model": "evaluated"}],
        }
        cfg = parse_eval_config(raw)
        assert not cfg.is_simple
        assert "evaluated" in cfg.services
        assert cfg.services["evaluated"].type == "vllm"

    def test_both_model_and_services_raises(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "services": {"s1": {"type": "api", "url": "http://x"}},
            "benchmarks": [{"name": "gsm8k"}],
        }
        with pytest.raises(Exception, match="not both"):
            parse_eval_config(raw)

    def test_no_model_or_services_raises(self):
        raw = {"benchmarks": [{"name": "gsm8k"}]}
        with pytest.raises(Exception):
            parse_eval_config(raw)

    def test_empty_benchmarks_raises(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "benchmarks": [],
        }
        with pytest.raises(Exception):
            parse_eval_config(raw)

    def test_cluster_defaults(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "benchmarks": [{"name": "gsm8k"}],
        }
        cfg = parse_eval_config(raw)
        assert cfg.cluster.type == "local"
        assert cfg.cluster.partition == "batch"

    def test_slurm_cluster(self):
        raw = {
            "model": {"url": "http://localhost:8000/v1", "id": "gpt-4"},
            "benchmarks": [{"name": "gsm8k"}],
            "cluster": {
                "type": "slurm",
                "hostname": "login-01",
                "partition": "gpu",
                "gres": "gpu:8",
                "walltime": "02:00:00",
            },
        }
        cfg = parse_eval_config(raw)
        assert cfg.cluster.type == "slurm"
        assert cfg.cluster.hostname == "login-01"
        assert cfg.cluster.gres == "gpu:8"

    def test_env_var_expansion(self, monkeypatch):
        monkeypatch.setenv("MODEL_URL", "http://my-server:8000/v1")
        monkeypatch.setenv("MODEL_ID", "llama-3")
        raw = {
            "model": {"url": "${MODEL_URL}", "id": "${MODEL_ID}"},
            "benchmarks": [{"name": "gsm8k"}],
        }
        cfg = parse_eval_config(raw)
        assert cfg.model.url == "http://my-server:8000/v1"
        assert cfg.model.id == "llama-3"

    def test_resolve_model_url_simple(self):
        cfg = EvalConfig(
            model=ModelConfig(url="http://localhost:8000/v1", id="gpt-4"),
            benchmarks=[BenchmarkConfig(name="gsm8k")],
        )
        assert cfg.resolve_model_url() == "http://localhost:8000/v1"

    def test_resolve_model_url_advanced(self):
        cfg = EvalConfig(
            services={
                "evaluated": ServiceConfig(type="api", url="http://remote:8000/v1"),
            },
            benchmarks=[BenchmarkConfig(name="gsm8k", model="evaluated")],
        )
        assert cfg.resolve_model_url("evaluated") == "http://remote:8000/v1"

    def test_managed_services(self):
        cfg = EvalConfig(
            services={
                "vllm_model": ServiceConfig(type="vllm", model="Qwen/Qwen3.5-9B"),
                "external": ServiceConfig(type="api", url="http://x"),
            },
            benchmarks=[BenchmarkConfig(name="gsm8k")],
        )
        managed = cfg.managed_services()
        assert "vllm_model" in managed
        assert "external" not in managed
