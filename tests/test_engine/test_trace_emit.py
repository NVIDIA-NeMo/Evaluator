# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ATIF-to-MLflow-Trace adapter (``_trace_emit``)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nemo_evaluator.engine.exporters import _trace_emit


class TestPureWalker:
    def test_iter_atif_steps_happy(self):
        traj = [{"steps": [{"source": "agent"}, {"source": "user"}]}]
        assert list(_trace_emit.iter_atif_steps(traj)) == [
            {"source": "agent"},
            {"source": "user"},
        ]

    def test_iter_atif_steps_multiple_docs(self):
        traj = [
            {"steps": [{"source": "user"}]},
            {"steps": [{"source": "agent"}]},
        ]
        assert [s["source"] for s in _trace_emit.iter_atif_steps(traj)] == ["user", "agent"]

    @pytest.mark.parametrize(
        "traj",
        [None, "not-a-list", 42, [], [None], [{"no_steps_key": 1}], [{"steps": None}], [{"steps": [None, "str", 7]}]],
    )
    def test_iter_atif_steps_defensive(self, traj):
        assert list(_trace_emit.iter_atif_steps(traj)) == []

    def test_classify_empty(self):
        assert _trace_emit._classify_trajectory([]) == "empty"
        assert _trace_emit._classify_trajectory(None) == "empty"
        assert _trace_emit._classify_trajectory([{"steps": []}]) == "empty"

    def test_classify_messages_only(self):
        traj = [{"steps": [{"source": "user", "message": "hi"}]}]
        assert _trace_emit._classify_trajectory(traj) == "messages"

    def test_classify_rich(self):
        traj = [
            {
                "steps": [
                    {"source": "user", "message": "hi"},
                    {"source": "agent", "tool_calls": [{"function_name": "Bash"}]},
                ]
            }
        ]
        assert _trace_emit._classify_trajectory(traj) == "rich"

    def test_trim_strings(self):
        assert _trace_emit._trim("short", 10) == "short"
        out = _trace_emit._trim("x" * 20, 5)
        assert out.startswith("xxxxx") and out.endswith("[truncated]")

    def test_trim_nested(self):
        out = _trace_emit._trim({"a": ["x" * 100, {"b": "x" * 100}]}, 5)
        assert out["a"][0].endswith("[truncated]")
        assert out["a"][1]["b"].endswith("[truncated]")

    def test_trim_passes_through_non_strings(self):
        assert _trace_emit._trim(42, 5) == 42
        assert _trace_emit._trim(None, 5) is None
        assert _trace_emit._trim(True, 5) is True

    def test_observation_map(self):
        step = {
            "observation": {
                "results": [
                    {"source_call_id": "a", "content": "hello"},
                    {"tool_call_id": "b", "content": {"nested": "json"}},
                ]
            }
        }
        out = _trace_emit._observation_map(step)
        assert out["a"] == "hello"
        assert '"nested"' in out["b"]

    @pytest.mark.parametrize(
        "step",
        [{}, {"observation": None}, {"observation": "str"}, {"observation": {"results": None}}],
    )
    def test_observation_map_defensive(self, step):
        assert _trace_emit._observation_map(step) == {}


class _FakeSpan:
    """Recording span for span-emission tests. Supports set_inputs/outputs/attributes."""

    def __init__(self, name: str, span_type: str):
        self.name = name
        self.span_type = span_type
        self.inputs: dict | None = None
        self.outputs: dict | None = None
        self.attributes: dict = {}

    def set_inputs(self, value):
        self.inputs = value

    def set_outputs(self, value):
        self.outputs = value

    def set_attributes(self, value):
        self.attributes.update(value)


class _FakeMLflow:
    """Captures ``start_span`` calls for verification."""

    def __init__(self):
        self.spans: list[_FakeSpan] = []

    def start_span(self, name: str, span_type: str):
        span = _FakeSpan(name, span_type)
        self.spans.append(span)

        class _Ctx:
            def __enter__(_self):
                return span

            def __exit__(_self, *args):
                return False

        return _Ctx()


@pytest.fixture
def fake_mlflow(monkeypatch):
    fake = _FakeMLflow()
    import sys

    mod = SimpleNamespace(start_span=fake.start_span)
    entities = SimpleNamespace(
        SpanType=SimpleNamespace(AGENT="AGENT", LLM="LLM", TOOL="TOOL", CHAIN="CHAIN"),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mod)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    return fake


class TestEmitTierA_Rich:
    def test_tool_call_emits_agent_llm_and_tool_spans(self, fake_mlflow):
        sample = {
            "problem_idx": 0,
            "repeat": 0,
            "reward": 1.0,
            "metadata": {"prompt": "solve this"},
            "model_response": "",
            "trajectory": [
                {
                    "steps": [
                        {"source": "user", "message": "solve this"},
                        {
                            "source": "agent",
                            "message": "Executed Bash tc_01",
                            "tool_calls": [
                                {"tool_call_id": "tc_01", "function_name": "Bash", "arguments": {"command": "ls"}},
                            ],
                            "observation": {"results": [{"source_call_id": "tc_01", "content": "a\nb"}]},
                        },
                    ]
                }
            ],
        }
        tier = _trace_emit.emit_sample_trace(sample, model_name="gpt", bench_name="pinchbench")
        assert tier == "rich"

        kinds = [(s.name, s.span_type) for s in fake_mlflow.spans]
        assert kinds[0] == ("sample_0", "AGENT")
        assert ("user", "CHAIN") in kinds
        assert ("Bash", "TOOL") in kinds
        assert not any(n == "assistant" for n, _ in kinds), "synthetic label should suppress empty LLM span"

        tool_span = next(s for s in fake_mlflow.spans if s.name == "Bash")
        assert tool_span.inputs == {"command": "ls"}
        assert tool_span.outputs == {"observation": "a\nb"}

        root = fake_mlflow.spans[0]
        assert root.attributes["nel.reward"] == 1.0
        assert root.attributes["nel.benchmark"] == "pinchbench"
        assert root.attributes["nel.problem_idx"] == 0
        assert root.outputs["reward"] == 1.0

    def test_large_content_truncated(self, fake_mlflow):
        sample = {
            "problem_idx": 1,
            "metadata": {"prompt": "x" * 9999},
            "trajectory": [
                {
                    "steps": [
                        {
                            "source": "agent",
                            "message": "",
                            "tool_calls": [
                                {"tool_call_id": "c", "function_name": "X", "arguments": {"blob": "y" * 9999}}
                            ],
                            "observation": {"results": [{"source_call_id": "c", "content": "z" * 9999}]},
                        }
                    ]
                }
            ],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b", content_max=100)
        tool = next(s for s in fake_mlflow.spans if s.name == "X")
        assert tool.inputs["blob"].endswith("[truncated]")
        assert tool.outputs["observation"].endswith("[truncated]")


class TestEmitTierB_MessagesOnly:
    def test_messages_only_no_tool_spans(self, fake_mlflow):
        sample = {
            "problem_idx": 2,
            "metadata": {"prompt": "hi"},
            "trajectory": [
                {
                    "steps": [
                        {"source": "user", "message": "hi"},
                        {"source": "agent", "message": "hello back", "tool_calls": []},
                    ]
                }
            ],
        }
        tier = _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="chat")
        assert tier == "messages"
        kinds = [(s.name, s.span_type) for s in fake_mlflow.spans]
        assert kinds[0] == ("sample_2", "AGENT")
        assert ("user", "CHAIN") in kinds
        assert ("assistant", "LLM") in kinds
        assert not any(t == "TOOL" for _, t in kinds)


class TestEmitTierC_ResponseOnly:
    def test_response_only_emits_singleshot_llm_span(self, fake_mlflow):
        sample = {
            "problem_idx": 3,
            "metadata": {"prompt": "What is 2+2?"},
            "model_response": "4",
            "tokens": 1,
            "latency_ms": 250,
            "reward": 1.0,
        }
        tier = _trace_emit.emit_sample_trace(sample, model_name="gpt", bench_name="gsm8k")
        assert tier == "response"

        kinds = [(s.name, s.span_type) for s in fake_mlflow.spans]
        assert kinds == [("sample_3", "AGENT"), ("llm", "LLM")]
        llm = fake_mlflow.spans[1]
        assert llm.inputs == {"prompt": "What is 2+2?"}
        assert llm.outputs == {"response": "4"}
        assert llm.attributes["nel.output_tokens"] == 1
        assert llm.attributes["nel.latency_ms"] == 250.0


class TestEmitTierD_MetadataOnly:
    def test_no_trajectory_no_response_emits_only_root(self, fake_mlflow):
        sample = {"problem_idx": 4, "reward": 0.0}
        tier = _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        assert tier == "meta"
        assert [s.name for s in fake_mlflow.spans] == ["sample_4"]


class TestEdgeCases:
    def test_missing_problem_idx_uses_generic_name(self, fake_mlflow):
        _trace_emit.emit_sample_trace({}, model_name="m", bench_name="b")
        assert fake_mlflow.spans[0].name == "sample"

    def test_repeat_appended_to_name(self, fake_mlflow):
        _trace_emit.emit_sample_trace({"problem_idx": 5, "repeat": 2}, model_name="m", bench_name="b")
        assert fake_mlflow.spans[0].name == "sample_5.r2"

    def test_non_dict_tool_call_skipped(self, fake_mlflow):
        sample = {
            "problem_idx": 0,
            "trajectory": [
                {
                    "steps": [
                        {
                            "source": "agent",
                            "message": "",
                            "tool_calls": ["not-a-dict", {"function_name": "X", "arguments": {}, "tool_call_id": "c"}],
                        }
                    ]
                }
            ],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        tool_spans = [s for s in fake_mlflow.spans if s.span_type == "TOOL"]
        assert len(tool_spans) == 1 and tool_spans[0].name == "X"

    def test_non_dict_arguments_wrapped(self, fake_mlflow):
        sample = {
            "problem_idx": 0,
            "trajectory": [
                {
                    "steps": [
                        {
                            "source": "agent",
                            "tool_calls": [{"function_name": "X", "arguments": "just a string", "tool_call_id": "c"}],
                        }
                    ]
                }
            ],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        tool = next(s for s in fake_mlflow.spans if s.name == "X")
        assert tool.inputs == {"value": "just a string"}

    def test_tool_call_without_observation(self, fake_mlflow):
        sample = {
            "problem_idx": 0,
            "trajectory": [
                {
                    "steps": [
                        {
                            "source": "agent",
                            "tool_calls": [{"function_name": "X", "tool_call_id": "cX", "arguments": {}}],
                            "observation": {},
                        }
                    ]
                }
            ],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        tool = next(s for s in fake_mlflow.spans if s.name == "X")
        assert tool.outputs is None


class TestPromptResolution:
    def test_prefers_top_level_prompt(self):
        s = {
            "prompt": "top level",
            "metadata": {"prompt": "meta"},
            "expected_answer": "ea",
            "trajectory": [{"steps": [{"source": "user", "message": "user msg"}]}],
        }
        assert _trace_emit._resolve_prompt(s) == "top level"

    def test_falls_back_to_metadata_prompt(self):
        s = {"metadata": {"prompt": "meta prompt"}, "expected_answer": "ea"}
        assert _trace_emit._resolve_prompt(s) == "meta prompt"

    def test_falls_back_to_first_user_step(self):
        s = {
            "metadata": {},
            "expected_answer": "ea",
            "trajectory": [
                {
                    "steps": [
                        {"source": "agent", "message": "hi"},
                        {"source": "user", "message": "real task"},
                    ]
                }
            ],
        }
        assert _trace_emit._resolve_prompt(s) == "real task"

    def test_falls_back_to_expected_answer(self):
        assert _trace_emit._resolve_prompt({"expected_answer": "42"}) == "42"

    def test_empty_when_nothing_available(self):
        assert _trace_emit._resolve_prompt({}) == ""

    def test_ignores_whitespace_only_values(self):
        s = {"prompt": "   ", "metadata": {"prompt": "meta"}}
        assert _trace_emit._resolve_prompt(s) == "meta"

    def test_tolerates_non_dict_metadata(self):
        assert _trace_emit._resolve_prompt({"metadata": "not a dict", "prompt": "p"}) == "p"


class TestRootSpanEnrichment:
    def test_scoring_details_and_scorer_rewards_on_root(self, fake_mlflow):
        sample = {
            "problem_idx": 7,
            "reward": 0.75,
            "prompt": "solve",
            "model_response": "answer",
            "scoring_details": {
                "judge": {"reason": "partially correct"},
                "scorer:llm_judge": {"reward": 0.75, "correct": False},
            },
            "scorer_rewards": {"scorer:llm_judge": 0.75, "scorer:exact": 0.0},
            "metadata": {"task_id": "t_memory", "task_name": "Memory", "category": "context"},
        }
        _trace_emit.emit_sample_trace(sample, model_name="gpt", bench_name="pinchbench")
        root = fake_mlflow.spans[0]
        assert root.attributes["nel.scoring_details"]["scorer:llm_judge"]["reward"] == 0.75
        assert root.attributes["nel.scorer.scorer:llm_judge"] == 0.75
        assert root.attributes["nel.scorer.scorer:exact"] == 0.0
        assert root.attributes["nel.task_id"] == "t_memory"
        assert root.attributes["nel.task_name"] == "Memory"
        assert root.attributes["nel.category"] == "context"

    def test_scoring_details_truncated(self, fake_mlflow):
        sample = {
            "problem_idx": 8,
            "reward": 0.0,
            "prompt": "p",
            "model_response": "r",
            "scoring_details": {"judge": {"reason": "x" * 9999}},
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b", content_max=50)
        root = fake_mlflow.spans[0]
        assert root.attributes["nel.scoring_details"]["judge"]["reason"].endswith("[truncated]")

    def test_no_scoring_details_no_attr(self, fake_mlflow):
        _trace_emit.emit_sample_trace(
            {"problem_idx": 9, "model_response": "x", "prompt": "y"},
            model_name="m",
            bench_name="b",
        )
        root = fake_mlflow.spans[0]
        assert "nel.scoring_details" not in root.attributes
        assert not any(k.startswith("nel.scorer.") for k in root.attributes)


class TestAgentOnlyTrajectorySynthesizesUserSpan:
    def test_synthesizes_user_span_for_agent_only_trajectory(self, fake_mlflow):
        sample = {
            "problem_idx": 10,
            "prompt": "Read the transcript and write a blog post.",
            "reward": 1.0,
            "trajectory": [
                {
                    "steps": [
                        {"source": "agent", "message": "Let me start by reading the file."},
                        {
                            "source": "agent",
                            "message": "",
                            "tool_calls": [
                                {"function_name": "Write", "tool_call_id": "c1", "arguments": {"path": "a"}}
                            ],
                            "observation": {"results": [{"source_call_id": "c1", "content": "ok"}]},
                        },
                    ]
                }
            ],
        }
        tier = _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="pinchbench")
        assert tier == "rich"

        names = [(s.name, s.span_type) for s in fake_mlflow.spans]
        assert names[0] == ("sample_10", "AGENT")
        assert names[1] == ("user", "CHAIN")
        user = fake_mlflow.spans[1]
        assert user.inputs == {"content": "Read the transcript and write a blog post."}

        root = fake_mlflow.spans[0]
        assert root.attributes.get("nel.user_prompt_synthesized") is True

    def test_no_synthesis_when_user_step_already_present(self, fake_mlflow):
        sample = {
            "problem_idx": 11,
            "prompt": "ignored because trajectory has user",
            "trajectory": [
                {
                    "steps": [
                        {"source": "user", "message": "from trajectory"},
                        {"source": "agent", "message": "ok"},
                    ]
                }
            ],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        user_spans = [s for s in fake_mlflow.spans if s.name == "user"]
        assert len(user_spans) == 1  # one from trajectory, none synthesized
        root = fake_mlflow.spans[0]
        assert "nel.user_prompt_synthesized" not in root.attributes

    def test_no_synthesis_when_no_prompt_available(self, fake_mlflow):
        sample = {
            "problem_idx": 12,
            "trajectory": [{"steps": [{"source": "agent", "message": "bare agent"}]}],
        }
        _trace_emit.emit_sample_trace(sample, model_name="m", bench_name="b")
        assert not any(s.name == "user" for s in fake_mlflow.spans)
