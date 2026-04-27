# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MLflow exporter tests: unit (fake mlflow module) + integration (real file store)."""

from __future__ import annotations

import os
import types
from pathlib import Path

import pytest


@pytest.fixture
def mlflow_fake(monkeypatch):
    """Replace the mlflow module imported by the exporter with an inert stub."""

    class _RunInfo:
        def __init__(self, **kwargs):
            self.experiment_id = kwargs.get("experiment_id", "exp1")
            self.run_id = kwargs.get("run_id") or "run1"

    class _RunCtx:
        def __init__(self, **kwargs):
            self.info = _RunInfo(**kwargs)

        def __enter__(self):
            return types.SimpleNamespace(info=self.info)

        def __exit__(self, *args):
            return False

    class _Experiment:
        def __init__(self):
            self.experiment_id = "exp1"

    class _SpanCtx:
        def __init__(self, name, span_type):
            self.name = name
            self.span_type = span_type

        def __enter__(self):
            span = types.SimpleNamespace(
                set_inputs=lambda _v: None,
                set_outputs=lambda _v: None,
                set_attributes=lambda _v: None,
            )
            return span

        def __exit__(self, *args):
            return False

    class _ML:
        set_tracking_uri = staticmethod(lambda *_: None)
        set_experiment = staticmethod(lambda *_: None)
        start_run = staticmethod(lambda *a, **k: _RunCtx(**k))
        start_span = staticmethod(lambda name, span_type=None: _SpanCtx(name, span_type))
        set_tags = staticmethod(lambda *_: None)
        set_tag = staticmethod(lambda *_: None)
        log_params = staticmethod(lambda *_: None)
        log_metrics = staticmethod(lambda *_: None)
        log_artifact = staticmethod(lambda *_, **__: None)
        log_artifacts = staticmethod(lambda *_, **__: None)
        get_experiment_by_name = staticmethod(lambda *_: None)
        search_runs = staticmethod(lambda **_: _EmptyDf())

    class _EmptyDf:
        empty = True

        def __len__(self) -> int:
            return 0

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://fake-mlflow")
    monkeypatch.setattr(
        "nemo_evaluator.engine.exporters.mlflow_export.MLFLOW_AVAILABLE",
        True,
        raising=True,
    )
    monkeypatch.setattr(
        "nemo_evaluator.engine.exporters.mlflow_export.mlflow",
        _ML,
        raising=True,
    )
    import sys

    monkeypatch.setitem(sys.modules, "mlflow", _ML)
    monkeypatch.setitem(
        sys.modules,
        "mlflow.entities",
        types.SimpleNamespace(
            SpanType=types.SimpleNamespace(AGENT="AGENT", LLM="LLM", TOOL="TOOL", CHAIN="CHAIN"),
        ),
    )
    return _ML, _RunCtx


# ── bundle factory (nel-next replacement for make_mlflow_job) ────────────


def _make_bundle(
    benchmark_name: str = "mmlu",
    *,
    run_id: str | None = None,
    scores: dict | None = None,
    config: dict | None = None,
    extra: dict | None = None,
) -> dict:
    """Build a minimal bundle matching the public artifact shape."""
    bundle: dict = {
        "run_id": run_id or f"eval-{benchmark_name}-000",
        "config_hash": "sha256:deadbeef",
        "sdk_version": "0.13.0",
        "timestamp": "2026-04-23T00:00:00+00:00",
        "config": config or {},
        "benchmark": {
            "name": benchmark_name,
            "samples": 10,
            "repeats": 1,
            "scores": scores if scores is not None else {"accuracy": 0.9},
        },
        "n_results": 10,
    }
    if extra:
        bundle.update(extra)
    return bundle


class TestTrackingURIValidation:
    def test_missing_tracking_uri_raises(self, monkeypatch):
        from pydantic import ValidationError

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        with pytest.raises(ValidationError, match="tracking_uri"):
            MLflowExporter()

    def test_tracking_uri_from_env(self, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://from-env")
        exp = MLflowExporter()
        assert exp._settings.tracking_uri == "http://from-env"

    def test_tracking_uri_indirection_via_env_var_name(self, monkeypatch):
        """When tracking_uri has no '://', treat it as an env var name."""
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        monkeypatch.setenv("MY_MLFLOW", "http://resolved-from-indirect")
        exp = MLflowExporter(tracking_uri="MY_MLFLOW")
        assert exp._settings.tracking_uri == "http://resolved-from-indirect"

    def test_explicit_tracking_uri_wins_over_env(self, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env-uri")
        exp = MLflowExporter(tracking_uri="http://explicit")
        assert exp._settings.tracking_uri == "http://explicit"


class TestExportHappyPath:
    def test_export_single_bundle(self, mlflow_fake):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        exp = MLflowExporter(tracking_uri="http://mlflow")
        bundle = _make_bundle("mmlu")
        exp.export([bundle])

    def test_export_logs_metrics(self, mlflow_fake, monkeypatch):
        """Scalar and dict-with-value scores both get logged as metrics."""
        from nemo_evaluator.engine.exporters import mlflow_export

        logged: dict = {}
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "log_metrics",
            staticmethod(lambda m: logged.update(m)),
            raising=False,
        )

        exp = mlflow_export.MLflowExporter(tracking_uri="http://mlflow")
        bundle = _make_bundle(
            "gsm8k",
            scores={
                "pass@1": {"value": 0.6},
                "accuracy": 0.9,
                "non_numeric": "skipme",
            },
        )
        exp.export([bundle])
        assert logged["pass_at_1"] == 0.6
        assert logged["accuracy"] == 0.9
        assert "non_numeric" not in logged

    def test_export_logs_tags_and_run_name(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        captured_tags: dict = {}
        captured_single_tags: list = []
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "set_tags",
            staticmethod(lambda t: captured_tags.update(t)),
            raising=False,
        )
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "set_tag",
            staticmethod(lambda k, v: captured_single_tags.append((k, v))),
            raising=False,
        )

        exp = mlflow_export.MLflowExporter(
            tracking_uri="http://mlflow",
            tags={"team": "frontier", "env": "dev"},
            description="Run description",
        )
        bundle = _make_bundle("gsm8k", run_id="eval-abcd1234", config={"foo": "bar"})
        exp.export([bundle])

        assert captured_tags["job_id"] == "eval-abcd1234"
        assert captured_tags["benchmark"] == "gsm8k"
        assert captured_tags["team"] == "frontier"
        assert ("mlflow.runName", "eval-eval-abcd1234") in captured_single_tags
        assert any(k == "mlflow.note.content" for k, _ in captured_single_tags)

    def test_export_custom_run_name_respected(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        single_tags: list = []
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "set_tag",
            staticmethod(lambda k, v: single_tags.append((k, v))),
            raising=False,
        )

        exp = mlflow_export.MLflowExporter(tracking_uri="http://mlflow", run_name="my-eval-run")
        exp.export([_make_bundle("mmlu")])
        assert ("mlflow.runName", "my-eval-run") in single_tags


class TestSkipExisting:
    def test_skip_existing_when_run_found(self, mlflow_fake, monkeypatch):
        """With skip_existing=True and a matching run, params/metrics are NOT logged."""
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        params_calls: list = []
        metrics_calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda *a, **k: params_calls.append((a, k))),
            raising=False,
        )
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_metrics",
            staticmethod(lambda *a, **k: metrics_calls.append((a, k))),
            raising=False,
        )

        exp = MLflowExporter(tracking_uri="http://mlflow", skip_existing=True)
        monkeypatch.setattr(exp, "_get_existing_run_id", lambda *a, **k: "existing-run-123")
        exp.export([_make_bundle("mmlu")])
        assert params_calls == []
        assert metrics_calls == []

    def test_update_existing_when_skip_off(self, mlflow_fake, monkeypatch):
        """With skip_existing=False and a matching run, metrics update; params don't."""
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        params_calls: list = []
        metrics_calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: params_calls.append(p)),
            raising=False,
        )
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_metrics",
            staticmethod(lambda m: metrics_calls.append(m)),
            raising=False,
        )

        exp = MLflowExporter(tracking_uri="http://mlflow", skip_existing=False)
        monkeypatch.setattr(exp, "_get_existing_run_id", lambda *a, **k: "existing-run-123")
        exp.export([_make_bundle("mmlu")])
        assert params_calls == []
        assert len(metrics_calls) == 1

    def test_no_existing_run_logs_params(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        params_calls: list = []
        metrics_calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: params_calls.append(p)),
            raising=False,
        )
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_metrics",
            staticmethod(lambda m: metrics_calls.append(m)),
            raising=False,
        )
        exp = MLflowExporter(tracking_uri="http://mlflow")
        monkeypatch.setattr(exp, "_get_existing_run_id", lambda *a, **k: None)
        exp.export([_make_bundle("mmlu")])
        assert len(params_calls) == 1
        assert len(metrics_calls) == 1


class TestLogConfigParams:
    def test_flattens_nested_config(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        logged: dict = {}
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: logged.update(p)),
            raising=False,
        )

        config = {
            "deployment": {"tensor_parallel_size": 8, "model": "test-model"},
            "evaluation": {
                "tasks": [
                    {"name": "task1", "config": {"param": "value1"}},
                    {"name": "task2"},
                ]
            },
        }
        exp = MLflowExporter(tracking_uri="http://mlflow", log_config_params=True)
        exp.export([_make_bundle("mmlu", config=config)])

        assert logged["config.deployment.tensor_parallel_size"] == "8"
        assert logged["config.deployment.model"] == "test-model"
        assert logged["config.evaluation.tasks.0.name"] == "task1"
        assert logged["config.evaluation.tasks.1.name"] == "task2"

    def test_max_depth_bounds_flattening(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        logged: dict = {}
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: logged.update(p)),
            raising=False,
        )

        exp = MLflowExporter(
            tracking_uri="http://mlflow",
            log_config_params=True,
            log_config_params_max_depth=2,
        )
        exp.export([_make_bundle("mmlu", config={"a": {"b": {"c": {"d": "deep"}}}})])

        assert "config.a.b" in logged
        assert "c" in logged["config.a.b"]
        assert "config.a.b.c" not in logged
        assert "config.a.b.c.d" not in logged

    def test_log_config_params_disabled_by_default(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        logged: dict = {}
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: logged.update(p)),
            raising=False,
        )

        exp = MLflowExporter(tracking_uri="http://mlflow")
        exp.export([_make_bundle("mmlu", config={"foo": "bar"})])
        assert not any(k.startswith("config.") for k in logged)


class TestExtraMetadata:
    def test_extra_metadata_is_logged_as_params(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        logged: dict = {}
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.mlflow.log_params",
            staticmethod(lambda p: logged.update(p)),
            raising=False,
        )

        exp = MLflowExporter(
            tracking_uri="http://mlflow",
            extra_metadata={"git_sha": "abc123", "cluster": "eos"},
        )
        exp.export([_make_bundle("mmlu")])
        assert logged["git_sha"] == "abc123"
        assert logged["cluster"] == "eos"


class TestAvailability:
    def test_raises_when_mlflow_missing(self, monkeypatch):
        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow")
        exp = MLflowExporter()
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters.mlflow_export.MLFLOW_AVAILABLE",
            False,
            raising=True,
        )
        with pytest.raises(ImportError, match="mlflow is required"):
            exp.export([_make_bundle("mmlu")])


class TestTraceEmissionWiring:
    """Exporter-side wiring of ``emit_traces`` + ``_emit_sample_traces``."""

    def test_emit_traces_on_by_default(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace",
            lambda sample, **kw: (calls.append((sample, kw)), "rich")[1],
        )

        bundle = _make_bundle("pinchbench", config={"model": "gpt-test"})
        bundle["_results"] = [
            {"problem_idx": 0, "reward": 1.0, "trajectory": [], "model_response": "ok"},
            {"problem_idx": 1, "reward": 0.0, "trajectory": [], "model_response": "no"},
        ]
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow").export([bundle])

        assert len(calls) == 2
        assert calls[0][0]["problem_idx"] == 0
        assert calls[0][1]["bench_name"] == "pinchbench"
        assert calls[0][1]["model_name"] == "gpt-test"

    def test_emit_traces_off_disables_emission(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace",
            lambda sample, **kw: calls.append(sample),
        )

        bundle = _make_bundle("x")
        bundle["_results"] = [{"problem_idx": 0}]
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow", emit_traces=False).export([bundle])

        assert calls == []

    def test_emit_traces_max_samples_cap(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace",
            lambda sample, **kw: (calls.append(sample), "meta")[1],
        )

        bundle = _make_bundle("x")
        bundle["_results"] = [{"problem_idx": i} for i in range(10)]
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow", emit_traces_max_samples=3).export([bundle])

        assert [s["problem_idx"] for s in calls] == [0, 1, 2]

    def test_emit_traces_skipped_for_existing_run(self, mlflow_fake, monkeypatch):
        """Re-exports to an existing run must NOT emit (would duplicate traces)."""
        from nemo_evaluator.engine.exporters import mlflow_export

        class _Iloc:
            def __getitem__(self, _i):
                return types.SimpleNamespace(run_id="prev-run")

        class _ExistingRunsDf:
            empty = False
            iloc = _Iloc()

            def __len__(self) -> int:
                return 1

        monkeypatch.setattr(
            mlflow_export.mlflow,
            "get_experiment_by_name",
            staticmethod(lambda *_: types.SimpleNamespace(experiment_id="exp1")),
            raising=False,
        )
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "search_runs",
            staticmethod(lambda **_: _ExistingRunsDf()),
            raising=False,
        )

        calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace",
            lambda sample, **kw: calls.append(sample),
        )

        bundle = _make_bundle("x", run_id="rx")
        bundle["_results"] = [{"problem_idx": 0}]
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow").export([bundle])

        assert calls == []

    def test_emit_tolerates_exceptions_per_sample(self, mlflow_fake, monkeypatch):
        """One bad sample must not abort the others nor fail the export."""
        from nemo_evaluator.engine.exporters import mlflow_export

        seen: list = []

        def fake_emit(sample, **kw):
            seen.append(sample["problem_idx"])
            if sample["problem_idx"] == 1:
                raise ValueError("oops")
            return "meta"

        monkeypatch.setattr("nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace", fake_emit)

        bundle = _make_bundle("x")
        bundle["_results"] = [{"problem_idx": i} for i in range(3)]
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow").export([bundle])

        assert seen == [0, 1, 2]

    def test_emit_noop_when_no_results(self, mlflow_fake, monkeypatch):
        from nemo_evaluator.engine.exporters import mlflow_export

        calls: list = []
        monkeypatch.setattr(
            "nemo_evaluator.engine.exporters._trace_emit.emit_sample_trace",
            lambda sample, **kw: calls.append(sample),
        )

        bundle = _make_bundle("x")
        mlflow_export.MLflowExporter(tracking_uri="http://mlflow").export([bundle])

        assert calls == []


class TestSanitize:
    def test_pass_at_rewrite(self):
        from nemo_evaluator.engine.exporters.mlflow_export import mlflow_sanitize

        assert mlflow_sanitize("pass@1", "metric") == "pass_at_1"
        assert mlflow_sanitize("pass@10", "metric") == "pass_at_10"

    def test_removes_invalid_key_chars(self):
        from nemo_evaluator.engine.exporters.mlflow_export import mlflow_sanitize

        out = mlflow_sanitize("foo!bar*baz/qux", "param_key")
        assert out == "foo_bar_baz/qux"

    def test_tag_value_strips_newlines(self):
        from nemo_evaluator.engine.exporters.mlflow_export import mlflow_sanitize

        assert mlflow_sanitize("a\nb\rc", "tag_value") == "a b c"

    def test_param_value_length_capped(self):
        from nemo_evaluator.engine.exporters.mlflow_export import mlflow_sanitize

        long = "x" * 1000
        assert len(mlflow_sanitize(long, "param_value")) == 250

    def test_tag_value_length_capped(self):
        from nemo_evaluator.engine.exporters.mlflow_export import mlflow_sanitize

        long = "x" * 10_000
        assert len(mlflow_sanitize(long, "tag_value")) == 5000


class TestArtifactPolicy:
    def test_excluded_patterns_match(self):
        from nemo_evaluator.engine.exporters.mlflow_export import (
            should_exclude_artifact,
        )

        assert should_exclude_artifact("response_stats_cache")
        assert should_exclude_artifact("lm_cache.db")
        assert should_exclude_artifact("something.lock")
        assert should_exclude_artifact("synthetic")
        assert should_exclude_artifact("debug.json")
        assert not should_exclude_artifact("results.jsonl")
        assert not should_exclude_artifact("eval-foo.json")

    def test_extra_patterns_extend_defaults(self):
        """User-supplied patterns extend (not replace) the hardcoded defaults."""
        from nemo_evaluator.engine.exporters.mlflow_export import (
            should_exclude_artifact,
        )

        # Defaults still apply when extra_patterns is provided
        assert should_exclude_artifact("response_stats_cache", extra_patterns=["deliverables"])
        # Extra exact-match pattern excludes
        assert should_exclude_artifact("deliverables", extra_patterns=["deliverables"])
        # Unrelated names still pass through
        assert not should_exclude_artifact("results.jsonl", extra_patterns=["deliverables"])

    def test_extra_patterns_glob_styles(self):
        """Extra patterns honor the same glob styles as EXCLUDED_PATTERNS."""
        from nemo_evaluator.engine.exporters.mlflow_export import (
            should_exclude_artifact,
        )

        assert should_exclude_artifact("my_deliverable_x", extra_patterns=["*deliverable*"])
        assert should_exclude_artifact("trace.parquet", extra_patterns=["*.parquet"])
        assert should_exclude_artifact("traces", extra_patterns=["traces"])
        assert not should_exclude_artifact("traces_dir", extra_patterns=["traces"])

    def test_extra_patterns_case_insensitive(self):
        """Extra patterns match case-insensitively, like the defaults."""
        from nemo_evaluator.engine.exporters.mlflow_export import (
            should_exclude_artifact,
        )

        assert should_exclude_artifact("Deliverables", extra_patterns=["deliverables"])
        assert should_exclude_artifact("deliverables", extra_patterns=["DELIVERABLES"])

    def test_get_copytree_ignore_with_extra_patterns(self):
        """get_copytree_ignore threads extra_patterns into the returned filter."""
        from nemo_evaluator.engine.exporters.mlflow_export import get_copytree_ignore

        ignore_func = get_copytree_ignore(["deliverables", "*.parquet"])
        contents = [
            "results.jsonl",
            "response_stats_cache",  # default exclusion
            "deliverables",
            "trace.parquet",
            "report.json",
        ]
        excluded = ignore_func("/some/dir", contents)
        assert "response_stats_cache" in excluded
        assert "deliverables" in excluded
        assert "trace.parquet" in excluded
        assert "results.jsonl" not in excluded
        assert "report.json" not in excluded

    def test_exporter_exclude_patterns_extends_defaults(self, mlflow_fake, tmp_path, monkeypatch):
        """exclude_patterns on the exporter extends defaults during recursive upload."""
        from nemo_evaluator.engine.exporters import mlflow_export

        bench = "mmlu"
        out_root = tmp_path / "out"
        task_dir = out_root / bench
        task_dir.mkdir(parents=True)
        (task_dir / "eval-mmlu-000.json").write_text('{"run_id":"x"}')
        (task_dir / "results.jsonl").write_text('{"problem_idx":0}\n')
        (task_dir / "deliverables").mkdir()
        (task_dir / "deliverables" / "out.txt").write_text("payload")
        (task_dir / "kept_dir").mkdir()
        (task_dir / "kept_dir" / "data.json").write_text("{}")
        # default exclusion still applies
        (task_dir / "response_stats_cache.db").write_text("cache")

        staged_contents: list[str] = []

        def fake_log_artifacts(local_dir, artifact_path=None):
            staged_contents.extend(sorted(os.listdir(local_dir)))

        monkeypatch.setattr(
            mlflow_export.mlflow,
            "log_artifacts",
            staticmethod(fake_log_artifacts),
            raising=False,
        )

        mlflow_export.MLflowExporter(
            tracking_uri="http://mlflow",
            only_required=False,
            copy_artifacts=True,
            exclude_patterns=["deliverables"],
        ).export(
            [_make_bundle(bench, run_id="eval-exclude-001")],
            config={"output_dir": str(out_root)},
        )

        # User-supplied pattern wins
        assert "deliverables" not in staged_contents
        # Default exclusion still applies (extends, doesn't replace)
        assert "response_stats_cache.db" not in staged_contents
        # Untouched files/dirs survive
        assert "kept_dir" in staged_contents
        assert "results.jsonl" in staged_contents

    def test_exporter_exclude_patterns_only_required_path(self, mlflow_fake, tmp_path, monkeypatch):
        """exclude_patterns also gates files in the only_required code path."""
        from nemo_evaluator.engine.exporters import mlflow_export

        bench = "mmlu"
        out_root = tmp_path / "out"
        task_dir = out_root / bench
        task_dir.mkdir(parents=True)
        (task_dir / "eval-mmlu-000.json").write_text('{"run_id":"x"}')
        (task_dir / "results.jsonl").write_text('{"problem_idx":0}\n')

        logged_paths: list[str] = []
        monkeypatch.setattr(
            mlflow_export.mlflow,
            "log_artifact",
            staticmethod(lambda p, artifact_path=None: logged_paths.append(p)),
            raising=False,
        )

        # Suppress results.jsonl via exclude_patterns even though it matches a required glob
        mlflow_export.MLflowExporter(
            tracking_uri="http://mlflow",
            only_required=True,
            copy_artifacts=True,
            exclude_patterns=["results.jsonl"],
        ).export(
            [_make_bundle(bench, run_id="eval-exclude-onlyreq-001")],
            config={"output_dir": str(out_root)},
        )

        names = [os.path.basename(p) for p in logged_paths]
        assert "results.jsonl" not in names
        assert "eval-mmlu-000.json" in names


mlflow_real = pytest.importorskip("mlflow", reason="mlflow not installed")


@pytest.fixture
def mlflow_file_tracking(tmp_path, monkeypatch):
    """Point the exporter at a file:// tracking store under tmp_path."""
    store = tmp_path / "mlruns"
    store.mkdir()
    uri = f"file://{store}"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    mlflow_real.set_tracking_uri(uri)
    return uri


def _make_output_dir(tmp_path: Path, benchmark_name: str) -> Path:
    task = tmp_path / "out" / benchmark_name
    task.mkdir(parents=True)
    (task / f"eval-{benchmark_name}-000.json").write_text('{"run_id":"test"}')
    (task / "results.jsonl").write_text('{"problem_idx":0}\n')
    (task / "trajectories.jsonl").write_text("trajectory-stub\n")
    (task / "response_stats_cache.db").write_text("cachefile")
    return tmp_path / "out"


class TestMLflowIntegration:
    def test_roundtrip_params_metrics_tags(self, mlflow_file_tracking, tmp_path):
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        output_dir = _make_output_dir(tmp_path, "gsm8k")

        exp = MLflowExporter(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-test",
            tags={"team": "frontier", "env": "ci"},
            extra_metadata={"git_sha": "abc123"},
            description="Round-trip integration test",
            log_config_params=True,
            copy_artifacts=True,
            only_required=True,
        )
        bundle = _make_bundle(
            "gsm8k",
            run_id="eval-integration-001",
            scores={"pass@1": {"value": 0.65}, "accuracy": 0.77},
            config={"model": "gpt-test", "temperature": 0.0},
        )
        exp.export([bundle], config={"output_dir": str(output_dir)})

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        experiment = client.get_experiment_by_name("integration-test")
        assert experiment is not None
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.job_id = 'eval-integration-001'",
        )
        assert len(runs) == 1
        run = runs[0]

        assert run.data.params["run_id"] == "eval-integration-001"
        assert run.data.params["benchmark"] == "gsm8k"
        assert run.data.params["git_sha"] == "abc123"
        assert run.data.params["config.model"] == "gpt-test"
        assert run.data.params["config.temperature"] == "0.0"

        assert run.data.metrics["pass_at_1"] == pytest.approx(0.65)
        assert run.data.metrics["accuracy"] == pytest.approx(0.77)

        assert run.data.tags["job_id"] == "eval-integration-001"
        assert run.data.tags["benchmark"] == "gsm8k"
        assert run.data.tags["team"] == "frontier"
        assert run.data.tags["env"] == "ci"
        assert run.data.tags["mlflow.runName"] == "eval-eval-integration-001"
        assert run.data.tags["mlflow.note.content"] == "Round-trip integration test"

        artifacts = client.list_artifacts(run.info.run_id, "gsm8k")
        names = sorted(a.path for a in artifacts)
        assert "gsm8k/eval-gsm8k-000.json" in names
        assert "gsm8k/results.jsonl" in names
        assert not any("trajectories" in n for n in names)
        assert not any("cache" in n for n in names)

    def test_copy_all_artifacts_excludes_cache_and_db(self, mlflow_file_tracking, tmp_path):
        """With only_required=False, everything is uploaded except EXCLUDED_PATTERNS entries."""
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        output_dir = _make_output_dir(tmp_path, "mmlu")
        exp = MLflowExporter(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-copy-all",
            only_required=False,
            copy_artifacts=True,
        )
        exp.export(
            [_make_bundle("mmlu", run_id="eval-copy-all-001")],
            config={"output_dir": str(output_dir)},
        )

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        exp_obj = client.get_experiment_by_name("integration-copy-all")
        runs = client.search_runs(
            [exp_obj.experiment_id],
            filter_string="tags.job_id = 'eval-copy-all-001'",
        )
        artifacts = client.list_artifacts(runs[0].info.run_id, "mmlu")
        names = [a.path for a in artifacts]
        assert any(n.endswith("eval-mmlu-000.json") for n in names)
        assert any(n.endswith("results.jsonl") for n in names)
        assert any(n.endswith("trajectories.jsonl") for n in names)
        assert not any("cache" in n for n in names)

    def test_skip_existing_on_rerun(self, mlflow_file_tracking, tmp_path):
        """Same bundle exported twice with skip_existing=True only creates one run."""
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        output_dir = _make_output_dir(tmp_path, "gsm8k")
        kwargs = dict(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-idempotent",
            skip_existing=True,
            only_required=True,
        )
        bundle = _make_bundle("gsm8k", run_id="eval-idempotent-001")

        MLflowExporter(**kwargs).export([bundle], config={"output_dir": str(output_dir)})
        MLflowExporter(**kwargs).export([bundle], config={"output_dir": str(output_dir)})

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        exp_obj = client.get_experiment_by_name("integration-idempotent")
        runs = client.search_runs(
            [exp_obj.experiment_id],
            filter_string="tags.job_id = 'eval-idempotent-001'",
        )
        assert len(runs) == 1

    def test_update_existing_run_refreshes_metrics(self, mlflow_file_tracking, tmp_path):
        """Re-export with skip_existing=False updates metrics on the same run."""
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        output_dir = _make_output_dir(tmp_path, "gsm8k")
        run_id = "eval-update-001"

        MLflowExporter(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-update",
            only_required=True,
        ).export(
            [_make_bundle("gsm8k", run_id=run_id, scores={"accuracy": 0.50})],
            config={"output_dir": str(output_dir)},
        )

        MLflowExporter(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-update",
            skip_existing=False,
            only_required=True,
        ).export(
            [_make_bundle("gsm8k", run_id=run_id, scores={"accuracy": 0.75})],
            config={"output_dir": str(output_dir)},
        )

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        exp_obj = client.get_experiment_by_name("integration-update")
        runs = client.search_runs(
            [exp_obj.experiment_id],
            filter_string=f"tags.job_id = '{run_id}'",
        )
        assert len(runs) == 1
        metric_history = client.get_metric_history(runs[0].info.run_id, "accuracy")
        values = [m.value for m in metric_history]
        assert 0.50 in values
        assert 0.75 in values

    def test_multiple_bundles_create_multiple_runs(self, mlflow_file_tracking, tmp_path):
        """One MLflow run per bundle."""
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.engine.exporters.mlflow_export import MLflowExporter

        output_dir = tmp_path / "out"
        output_dir.mkdir()
        for bench in ("mmlu", "gsm8k", "humaneval"):
            _make_output_dir(tmp_path, bench)

        bundles = [
            _make_bundle("mmlu", run_id="eval-mmlu-001"),
            _make_bundle("gsm8k", run_id="eval-gsm8k-001"),
            _make_bundle("humaneval", run_id="eval-humaneval-001"),
        ]
        MLflowExporter(
            tracking_uri=mlflow_file_tracking,
            experiment_name="integration-multi",
            only_required=True,
        ).export(bundles, config={"output_dir": str(output_dir)})

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        exp_obj = client.get_experiment_by_name("integration-multi")
        runs = client.search_runs([exp_obj.experiment_id])
        assert len(runs) == 3
        job_ids = {r.data.tags.get("job_id") for r in runs}
        assert job_ids == {
            "eval-mmlu-001",
            "eval-gsm8k-001",
            "eval-humaneval-001",
        }


class TestMLflowCLIIntegration:
    """End-to-end: ``nel export --dest mlflow <run-dir>`` writes to a real file store.

    Covers the post-hoc CLI path (not just direct exporter use) and guards against
    the class of bugs where the CLI computes the wrong ``output_dir`` for the
    exporter — e.g. passing the *bench dir* and causing artifact paths like
    ``<run>/<bench>/<bench>/results.jsonl`` instead of ``<run>/<bench>/results.jsonl``.
    """

    def _make_run_tree(self, tmp_path: Path, bench: str, run_id: str, scores: dict | None = None) -> Path:
        """Build a plausible ``nel eval`` output directory on disk."""
        import json as _json

        run = tmp_path / "nmp_out"
        bench_dir = run / bench
        bench_dir.mkdir(parents=True)
        bundle = _make_bundle(bench, run_id=run_id, scores=scores, config={"model": "gpt-test"})
        (bench_dir / f"{run_id}.json").write_text(_json.dumps(bundle))
        (bench_dir / "results.jsonl").write_text(
            "\n".join(_json.dumps({"problem_idx": i, "reward": 1.0 if i < 2 else 0.0}) for i in range(3)) + "\n"
        )
        return run

    def test_cli_export_logs_run_to_real_mlflow(self, mlflow_file_tracking, tmp_path):
        from click.testing import CliRunner
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.cli.export import export_cmd

        run = self._make_run_tree(tmp_path, "agentic-use", "eval-cli-e2e-001")

        result = CliRunner().invoke(
            export_cmd,
            [
                str(run),
                "--dest",
                "mlflow",
                "-o",
                f"tracking_uri={mlflow_file_tracking}",
                "-o",
                "experiment_name=integration-cli",
                "-o",
                "only_required=true",
                "-o",
                "copy_artifacts=true",
            ],
        )
        assert result.exit_code == 0, result.output

        client = MlflowClient(tracking_uri=mlflow_file_tracking)
        experiment = client.get_experiment_by_name("integration-cli")
        assert experiment is not None, "CLI did not create the experiment"
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="tags.job_id = 'eval-cli-e2e-001'",
        )
        assert len(runs) == 1, f"expected 1 run, got {len(runs)}"
        logged = runs[0]
        assert logged.data.tags["benchmark"] == "agentic-use"
        assert "accuracy" in logged.data.metrics

        artifact_paths = {a.path for a in client.list_artifacts(logged.info.run_id, "agentic-use")}
        assert "agentic-use/eval-cli-e2e-001.json" in artifact_paths
        assert "agentic-use/results.jsonl" in artifact_paths
        assert not any("agentic-use/agentic-use" in p for p in artifact_paths), (
            f"CLI duplicated the bench-name segment in artifact paths: {artifact_paths}"
        )

    def test_cli_export_from_bench_dir_does_not_duplicate(self, mlflow_file_tracking, tmp_path):
        """Pointing the CLI at the bench dir (or a bundle file) must behave identically.

        Regression for the ``_output_path``-vs-``config['output_dir']`` mix-up
        that produced ``<run>/<bench>/<bench>/...`` paths on re-export.
        """
        from click.testing import CliRunner
        from mlflow.tracking import MlflowClient

        from nemo_evaluator.cli.export import export_cmd

        run = self._make_run_tree(tmp_path, "gsm8k", "eval-cli-bench-dir-001")
        bench_dir = run / "gsm8k"

        for target in (bench_dir, bench_dir / "eval-cli-bench-dir-001.json"):
            result = CliRunner().invoke(
                export_cmd,
                [
                    str(target),
                    "--dest",
                    "mlflow",
                    "-o",
                    f"tracking_uri={mlflow_file_tracking}",
                    "-o",
                    f"experiment_name=integration-cli-{target.name}",
                    "-o",
                    "only_required=true",
                    "-o",
                    "copy_artifacts=true",
                    "-o",
                    "skip_existing=false",
                ],
            )
            assert result.exit_code == 0, result.output

            client = MlflowClient(tracking_uri=mlflow_file_tracking)
            exp = client.get_experiment_by_name(f"integration-cli-{target.name}")
            assert exp is not None
            logged = client.search_runs(
                [exp.experiment_id],
                filter_string="tags.job_id = 'eval-cli-bench-dir-001'",
            )[0]
            paths = {a.path for a in client.list_artifacts(logged.info.run_id, "gsm8k")}
            assert "gsm8k/results.jsonl" in paths
            assert not any("gsm8k/gsm8k" in p for p in paths), paths
