"""Tests for config YAML schema validation and env var expansion."""
import os
import pytest
from nemo_evaluator.config_schema import parse_config, _expand_env


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


class TestParseConfig:
    def test_nested_format(self):
        raw = {
            "evaluation": {
                "model_url": "http://example.com",
                "model_id": "gpt-4",
                "tasks": [{"benchmark": "gsm8k"}],
            }
        }
        cfg = parse_config(raw)
        assert cfg.model_url == "http://example.com"
        assert len(cfg.tasks) == 1
        assert cfg.tasks[0].benchmark == "gsm8k"

    def test_flat_format(self):
        raw = {
            "model_url": "http://example.com",
            "tasks": [{"benchmark": "gsm8k", "repeats": 4}],
        }
        cfg = parse_config(raw)
        assert cfg.model_url == "http://example.com"
        assert cfg.tasks[0].repeats == 4

    def test_harness_task(self):
        raw = {
            "evaluation": {
                "tasks": [
                    {"harness": "lm-eval", "tasks": ["gsm8k"]},
                    {"harness": "lm-eval", "tasks": ["aime25"]},
                ]
            }
        }
        cfg = parse_config(raw)
        assert len(cfg.tasks) == 2
        assert cfg.tasks[0].harness == "lm-eval"
        assert cfg.tasks[1].tasks == ["aime25"]

    def test_empty_tasks_raises(self):
        with pytest.raises(Exception, match="at least one task"):
            parse_config({"evaluation": {"tasks": []}})

    def test_task_without_target_raises(self):
        with pytest.raises(Exception, match="at least one of"):
            parse_config({"evaluation": {"tasks": [{"repeats": 4}]}})

    def test_mixed_tasks(self):
        raw = {
            "evaluation": {
                "tasks": [
                    {"benchmark": "gsm8k", "repeats": 4},
                    {"adapter": "gym://localhost:9090"},
                    {"harness": "lm-eval", "tasks": ["aime25"], "fewshot": 0},
                ]
            }
        }
        cfg = parse_config(raw)
        assert len(cfg.tasks) == 3
        assert cfg.tasks[0].benchmark == "gsm8k"
        assert cfg.tasks[1].adapter == "gym://localhost:9090"
        assert cfg.tasks[2].harness == "lm-eval"
