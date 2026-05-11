# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``nel export`` — post-hoc export CLI."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

from nemo_evaluator.cli.export import _parse_override, _parse_value, export_cmd


@pytest.fixture
def run_tree(tmp_path):
    gsm_dir = tmp_path / "run" / "gsm8k"
    gsm_dir.mkdir(parents=True)
    bundle = {
        "run_id": "eval-cli-gsm8k-001",
        "config_hash": "sha256:deadbeef",
        "sdk_version": "0.13.0",
        "timestamp": "2026-04-23T00:00:00+00:00",
        "config": {"model": "test-model", "temperature": 0.0},
        "benchmark": {
            "name": "gsm8k",
            "samples": 3,
            "repeats": 1,
            "scores": {"pass@1": {"value": 0.67}},
        },
        "n_results": 3,
    }
    (gsm_dir / f"{bundle['run_id']}.json").write_text(json.dumps(bundle))
    (gsm_dir / "results.jsonl").write_text(
        "\n".join(json.dumps({"problem_idx": i, "reward": 1 if i < 2 else 0}) for i in range(3)) + "\n"
    )
    return tmp_path / "run"


class TestOverrideParsing:
    def test_parse_value_json(self):
        assert _parse_value("42") == 42
        assert _parse_value("3.14") == 3.14
        assert _parse_value("true") is True
        assert _parse_value("false") is False
        assert _parse_value("null") is None
        assert _parse_value('{"a": 1}') == {"a": 1}
        assert _parse_value("[1,2,3]") == [1, 2, 3]

    def test_parse_value_falls_back_to_string(self):
        assert _parse_value("http://mlflow:5000") == "http://mlflow:5000"
        assert _parse_value("my-exp") == "my-exp"

    def test_parse_override_happy(self):
        assert _parse_override("tracking_uri=http://mlflow") == (
            "tracking_uri",
            "http://mlflow",
        )
        assert _parse_override("skip_existing=true") == ("skip_existing", True)
        assert _parse_override('tags={"team":"x"}') == (
            "tags",
            {"team": "x"},
        )

    def test_parse_override_bad_shape_raises(self):
        import click

        with pytest.raises(click.BadParameter):
            _parse_override("no-equals-sign")
        with pytest.raises(click.BadParameter):
            _parse_override("=value")


class TestCLI:
    def _run(self, run_tree, *args) -> "CliRunner.invoke":
        runner = CliRunner()
        return runner.invoke(export_cmd, [str(run_tree), *args])

    def test_missing_dest_errors(self, run_tree):
        runner = CliRunner()
        result = runner.invoke(export_cmd, [str(run_tree)])
        assert result.exit_code != 0
        assert "Missing option" in result.output or "Usage" in result.output

    def test_unknown_exporter_errors(self, run_tree, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://x")
        result = self._run(run_tree, "--dest", "nonexistent-exporter")
        assert result.exit_code != 0
        assert "Unknown exporter" in result.output or "nonexistent" in result.output

    def test_no_bundles_errors(self, tmp_path):
        runner = CliRunner()
        empty = tmp_path / "empty-run"
        empty.mkdir()
        result = runner.invoke(export_cmd, [str(empty), "--dest", "mlflow"])
        assert result.exit_code != 0
        assert "No eval-*.json bundle files" in result.output

    def test_calls_exporter_with_overrides(self, run_tree, monkeypatch):
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def export(self, bundles, config=None):
                captured["bundles"] = bundles
                captured["config"] = config

        def fake_get_exporter(name, **kwargs):
            assert name == "mlflow"
            return _FakeExporter(**kwargs)

        monkeypatch.setattr("nemo_evaluator.engine.exporters.get_exporter", fake_get_exporter)

        result = self._run(
            run_tree,
            "--dest",
            "mlflow",
            "-o",
            "tracking_uri=http://mlflow:5000",
            "-o",
            "experiment_name=my-exp",
            "-o",
            "skip_existing=true",
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"] == {
            "tracking_uri": "http://mlflow:5000",
            "experiment_name": "my-exp",
            "skip_existing": True,
        }
        assert len(captured["bundles"]) == 1
        assert len(captured["bundles"][0]["_results"]) == 3
        # output_dir must be the *run dir* (parent of the bench dir), because
        # exporters append ``/<bench_name>`` themselves. Passing the bench dir
        # here would produce duplicated ``run/gsm8k/gsm8k/...`` output paths.
        assert captured["config"]["output_dir"].endswith("/run")
        assert not captured["config"]["output_dir"].endswith("/gsm8k")

    def test_output_dir_override(self, run_tree, monkeypatch, tmp_path):
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                pass

            def export(self, bundles, config=None):
                captured["config"] = config

        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.get_exporter",
            lambda name, **kw: _FakeExporter(**kw),
        )
        override = tmp_path / "custom-out"
        override.mkdir()

        result = self._run(
            run_tree,
            "--dest",
            "mlflow",
            "-o",
            "tracking_uri=http://x",
            "--output-dir",
            str(override),
        )
        assert result.exit_code == 0, result.output
        assert captured["config"]["output_dir"] == str(override)

    def test_config_file_merged_before_overrides(self, run_tree, tmp_path, monkeypatch):
        """``-o`` should win over values from --config."""
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def export(self, bundles, config=None):
                pass

        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.get_exporter",
            lambda name, **kw: _FakeExporter(**kw),
        )

        cfg = tmp_path / "exporter.json"
        cfg.write_text(
            json.dumps(
                {
                    "tracking_uri": "http://from-file",
                    "experiment_name": "from-file",
                    "extra_metadata": {"git_sha": "abc"},
                }
            )
        )
        result = self._run(
            run_tree,
            "--dest",
            "mlflow",
            "--config",
            str(cfg),
            "-o",
            "experiment_name=from-override",
        )
        assert result.exit_code == 0, result.output
        assert captured["kwargs"]["tracking_uri"] == "http://from-file"
        assert captured["kwargs"]["experiment_name"] == "from-override"
        assert captured["kwargs"]["extra_metadata"] == {"git_sha": "abc"}

    def test_config_file_with_nested_exporter_block(self, run_tree, tmp_path, monkeypatch):
        """Config may be ``{mlflow: {...}, inspect: {...}}``; only matching block is used."""
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                captured["kwargs"] = kwargs

            def export(self, bundles, config=None):
                pass

        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.get_exporter",
            lambda name, **kw: _FakeExporter(**kw),
        )

        cfg = tmp_path / "exporter.json"
        cfg.write_text(
            json.dumps(
                {
                    "mlflow": {"tracking_uri": "http://ml", "experiment_name": "ml-exp"},
                    "inspect": {"format": "eval"},
                }
            )
        )
        result = self._run(run_tree, "--dest", "mlflow", "--config", str(cfg))
        assert result.exit_code == 0, result.output
        assert captured["kwargs"] == {
            "tracking_uri": "http://ml",
            "experiment_name": "ml-exp",
        }

    def test_run_root_expands_to_all_benchmarks(self, tmp_path, monkeypatch):
        """``nel export <run-root>`` picks up bundles under each benchmark subdir."""
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                pass

            def export(self, bundles, config=None):
                captured["bundles"] = bundles

        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.get_exporter",
            lambda name, **kw: _FakeExporter(**kw),
        )

        run = tmp_path / "multi-run"
        for bench in ("gsm8k", "mmlu"):
            d = run / bench
            d.mkdir(parents=True)
            (d / f"eval-{bench}-001.json").write_text(
                json.dumps(
                    {
                        "run_id": f"eval-{bench}-001",
                        "benchmark": {"name": bench, "scores": {"accuracy": 0.5}},
                        "config": {},
                    }
                )
            )

        runner = CliRunner()
        result = runner.invoke(
            export_cmd,
            [str(run), "--dest", "mlflow", "-o", "tracking_uri=http://x"],
        )
        assert result.exit_code == 0, result.output
        benchmark_names = {b["benchmark"]["name"] for b in captured["bundles"]}
        assert benchmark_names == {"gsm8k", "mmlu"}

    def test_output_dir_is_run_dir_not_bench_dir(self, tmp_path, monkeypatch):
        """Regression: ``output_dir`` must be the run dir (parent of the bench
        dirs), never the bench dir itself. Otherwise the exporter's own
        ``output_dir / <bench_name>`` suffixing produces duplicated paths like
        ``run/foo/foo/foo_inspect.eval``."""
        captured = {}

        class _FakeExporter:
            def __init__(self, **kwargs):
                pass

            def export(self, bundles, config=None):
                captured["config"] = config

        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.get_exporter",
            lambda name, **kw: _FakeExporter(**kw),
        )

        run = tmp_path / "some-run"
        bench_dir = run / "agentic-use"
        bench_dir.mkdir(parents=True)
        (bench_dir / "eval-agentic-use-001.json").write_text(
            json.dumps(
                {
                    "run_id": "eval-agentic-use-001",
                    "benchmark": {"name": "agentic-use", "scores": {"pass@1": 1.0}},
                    "config": {},
                }
            )
        )

        runner = CliRunner()
        cases = [
            [str(run), "--dest", "mlflow", "-o", "tracking_uri=http://x"],
            [str(bench_dir), "--dest", "mlflow", "-o", "tracking_uri=http://x"],
            [
                str(bench_dir / "eval-agentic-use-001.json"),
                "--dest",
                "mlflow",
                "-o",
                "tracking_uri=http://x",
            ],
        ]
        for args in cases:
            captured.clear()
            result = runner.invoke(export_cmd, args)
            assert result.exit_code == 0, result.output
            od = captured["config"]["output_dir"]
            assert not od.endswith("agentic-use"), f"output_dir should be the run dir, got bench dir: {od}"
            assert od.endswith("some-run"), f"expected run dir, got: {od}"
