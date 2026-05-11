# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ATIF v1.6 compliance tests for nel-next trajectory builders."""

from __future__ import annotations

import copy

import pytest

from nemo_evaluator.solvers.trajectory_util import (
    build_atif_trajectory,
    build_single_turn_atif,
)

from .atif_validator import (
    ATIFValidationError,
    assert_atif_compliant,
    is_valid_atif,
    validate_atif,
)


def _good_doc() -> dict:
    return {
        "schema_version": "ATIF-v1.6",
        "session_id": "025B810F-B3A2-4C67-93C0-FE7A142A947A",
        "agent": {"name": "harbor-agent", "version": "1.0.0", "model_name": "gemini-2.5-flash"},
        "steps": [
            {
                "step_id": 1,
                "timestamp": "2025-10-11T10:30:00Z",
                "source": "user",
                "message": "What is the current trading price of Alphabet (GOOGL)?",
            },
            {
                "step_id": 2,
                "timestamp": "2025-10-11T10:30:02Z",
                "source": "agent",
                "model_name": "gemini-2.5-flash",
                "message": "I will search for the current trading price and volume for GOOGL.",
                "tool_calls": [
                    {
                        "tool_call_id": "call_price_1",
                        "function_name": "financial_search",
                        "arguments": {"ticker": "GOOGL", "metric": "price"},
                    }
                ],
                "observation": {
                    "results": [{"source_call_id": "call_price_1", "content": "GOOGL is trading at $185.35"}]
                },
                "metrics": {"prompt_tokens": 520, "completion_tokens": 80, "cached_tokens": 200},
            },
        ],
        "final_metrics": {"total_prompt_tokens": 520, "total_completion_tokens": 80, "total_steps": 2},
    }


class TestHappyPath:
    def test_rfc_example_validates_strict(self):
        assert validate_atif(_good_doc(), strict=True) == []
        assert is_valid_atif(_good_doc(), strict=True) is True

    def test_accepts_list_wrapper(self):
        assert validate_atif([_good_doc()]) == []

    def test_minimum_viable_trajectory(self):
        doc = {
            "schema_version": "ATIF-v1.6",
            "session_id": "x",
            "agent": {"name": "a", "version": "1"},
            "steps": [{"step_id": 1, "source": "user", "message": "hi"}],
        }
        assert validate_atif(doc, strict=True) == []

    def test_empty_message_string_is_allowed(self):
        """RFC: 'This field is required but can be an empty string.'"""
        doc = _good_doc()
        doc["steps"].append({"step_id": 3, "source": "agent", "message": ""})
        assert validate_atif(doc, strict=True) == []

    def test_assert_helper_no_error_on_valid(self):
        assert_atif_compliant(_good_doc(), strict=True)


class TestRootLevelViolations:
    @pytest.mark.parametrize("field", ["schema_version", "session_id", "agent", "steps"])
    def test_missing_required_field(self, field):
        doc = _good_doc()
        del doc[field]
        errors = validate_atif(doc)
        assert any(f"missing required field '{field}'" in e for e in errors), errors

    def test_schema_version_must_start_with_atif(self):
        doc = _good_doc()
        doc["schema_version"] = "v1.6"
        errors = validate_atif(doc)
        assert any("must start with 'ATIF-'" in e for e in errors), errors

    def test_empty_session_id_rejected(self):
        doc = _good_doc()
        doc["session_id"] = ""
        errors = validate_atif(doc)
        assert any("session_id: must be non-empty" in e for e in errors), errors

    def test_empty_steps_rejected(self):
        doc = _good_doc()
        doc["steps"] = []
        errors = validate_atif(doc)
        assert any("steps: must be non-empty" in e for e in errors), errors

    def test_wrong_root_type(self):
        assert "expected dict" in validate_atif("not a doc")[0]
        assert "single-document list wrapper" in validate_atif([1, 2])[0]


class TestAgentSchema:
    @pytest.mark.parametrize("field", ["name", "version"])
    def test_agent_missing_required(self, field):
        doc = _good_doc()
        del doc["agent"][field]
        errors = validate_atif(doc)
        assert any(f"root.agent.{field}" in e for e in errors), errors

    def test_agent_empty_name_rejected(self):
        doc = _good_doc()
        doc["agent"]["name"] = ""
        errors = validate_atif(doc)
        assert any("root.agent.name" in e for e in errors), errors

    def test_agent_tool_definitions_validated(self):
        doc = _good_doc()
        doc["agent"]["tool_definitions"] = [{"type": "not_function", "function": {}}]
        errors = validate_atif(doc)
        assert any("tool_definitions[0].type" in e for e in errors), errors


class TestStepViolations:
    def test_non_sequential_step_ids(self):
        doc = _good_doc()
        doc["steps"][1]["step_id"] = 99
        errors = validate_atif(doc)
        assert any("step_id: expected sequential ordinal 2" in e for e in errors), errors

    def test_string_step_id_rejected(self):
        doc = _good_doc()
        doc["steps"][0]["step_id"] = "1"
        errors = validate_atif(doc)
        assert any("step_id: required int" in e for e in errors), errors

    def test_invalid_source_enum(self):
        doc = _good_doc()
        doc["steps"][0]["source"] = "assistant"
        errors = validate_atif(doc)
        assert any("source: must be one of" in e for e in errors), errors

    def test_missing_message_rejected(self):
        doc = _good_doc()
        del doc["steps"][0]["message"]
        errors = validate_atif(doc)
        assert any("message: required field missing" in e for e in errors), errors

    def test_non_string_non_list_message_rejected(self):
        doc = _good_doc()
        doc["steps"][0]["message"] = 42
        errors = validate_atif(doc)
        assert any("message: expected string or list" in e for e in errors), errors

    def test_bad_timestamp_rejected(self):
        doc = _good_doc()
        doc["steps"][0]["timestamp"] = "10/11/2025"
        errors = validate_atif(doc)
        assert any("timestamp: not a valid ISO 8601" in e for e in errors), errors

    @pytest.mark.parametrize("field", ["model_name", "reasoning_effort", "reasoning_content", "tool_calls", "metrics"])
    def test_agent_only_fields_rejected_on_user_step(self, field):
        doc = _good_doc()
        doc["steps"][0][field] = "x" if field != "tool_calls" else []
        errors = validate_atif(doc)
        assert any(f".{field}: only allowed on agent steps" in e for e in errors), errors


class TestToolCallViolations:
    def _step_with_tool_calls(self, **overrides):
        call = {
            "tool_call_id": "t1",
            "function_name": "my_fn",
            "arguments": {"x": 1},
        }
        call.update(overrides)
        doc = _good_doc()
        doc["steps"][1]["tool_calls"] = [call]
        doc["steps"][1]["observation"] = {"results": []}
        return doc

    def test_missing_tool_call_id(self):
        errors = validate_atif(self._step_with_tool_calls(tool_call_id=""))
        assert any("tool_call_id: required non-empty string" in e for e in errors), errors

    def test_missing_function_name(self):
        errors = validate_atif(self._step_with_tool_calls(function_name=""))
        assert any("function_name: required non-empty string" in e for e in errors), errors

    def test_non_dict_arguments(self):
        errors = validate_atif(self._step_with_tool_calls(arguments="raw"))
        assert any("arguments: required dict" in e for e in errors), errors

    def test_empty_arguments_allowed(self):
        assert validate_atif(self._step_with_tool_calls(arguments={})) == []

    def test_duplicate_tool_call_ids_rejected(self):
        doc = _good_doc()
        doc["steps"][1]["tool_calls"] = [
            {"tool_call_id": "dup", "function_name": "a", "arguments": {}},
            {"tool_call_id": "dup", "function_name": "b", "arguments": {}},
        ]
        errors = validate_atif(doc)
        assert any("duplicate id 'dup'" in e for e in errors), errors


class TestObservationViolations:
    def test_results_required(self):
        doc = _good_doc()
        doc["steps"][1]["observation"] = {}
        errors = validate_atif(doc)
        assert any("observation.results: required list" in e for e in errors), errors

    def test_source_call_id_must_reference_tool_call(self):
        doc = _good_doc()
        doc["steps"][1]["observation"] = {"results": [{"source_call_id": "phantom_id", "content": "..."}]}
        errors = validate_atif(doc)
        assert any("source_call_id: 'phantom_id' does not reference any known tool_call_id" in e for e in errors), (
            errors
        )

    def test_null_source_call_id_allowed_for_non_tool_actions(self):
        doc = _good_doc()
        doc["steps"][1].pop("tool_calls")
        doc["steps"][1]["observation"] = {"results": [{"content": "raw action output"}]}
        assert validate_atif(doc) == []

    def test_subagent_trajectory_ref_requires_session_id(self):
        doc = _good_doc()
        doc["steps"][1].pop("tool_calls")
        doc["steps"][1]["observation"] = {"results": [{"subagent_trajectory_ref": [{"trajectory_path": "s3://x"}]}]}
        errors = validate_atif(doc)
        assert any("session_id: required non-empty string" in e for e in errors), errors


class TestMultimodalContent:
    def test_valid_text_and_image_content_parts(self):
        doc = _good_doc()
        doc["steps"][0]["message"] = [
            {"type": "text", "text": "What is in this image?"},
            {
                "type": "image",
                "source": {"media_type": "image/png", "path": "images/q.png"},
            },
        ]
        assert validate_atif(doc) == []

    def test_image_part_rejects_text_field(self):
        doc = _good_doc()
        doc["steps"][0]["message"] = [
            {"type": "image", "text": "stray", "source": {"media_type": "image/png", "path": "p"}}
        ]
        errors = validate_atif(doc)
        assert any("must be omitted when type='image'" in e for e in errors), errors

    def test_unsupported_media_type_rejected(self):
        doc = _good_doc()
        doc["steps"][0]["message"] = [{"type": "image", "source": {"media_type": "image/bmp", "path": "p"}}]
        errors = validate_atif(doc)
        assert any("media_type: must be one of" in e for e in errors), errors


class TestMetrics:
    def test_cached_tokens_exceeding_prompt_tokens_rejected(self):
        doc = _good_doc()
        doc["steps"][1]["metrics"] = {"prompt_tokens": 10, "cached_tokens": 20}
        errors = validate_atif(doc)
        assert any("cached_tokens (20) exceeds prompt_tokens" in e for e in errors), errors

    def test_wrong_types_rejected(self):
        doc = _good_doc()
        doc["steps"][1]["metrics"] = {"prompt_tokens": "520"}
        errors = validate_atif(doc)
        assert any("prompt_tokens: expected int" in e for e in errors), errors


class TestCompletenessStrict:
    def test_agent_only_trajectory_strict_violation(self):
        doc = {
            "schema_version": "ATIF-v1.6",
            "session_id": "x",
            "agent": {"name": "openclaw", "version": "1.0.0"},
            "steps": [
                {"step_id": 1, "source": "agent", "message": "The answer is 42."},
                {"step_id": 2, "source": "agent", "message": "Saved to answer.txt."},
            ],
        }
        assert validate_atif(doc, strict=False) == []
        errors = validate_atif(doc, strict=True)
        assert any("trajectory has agent activity but no user or system step" in e for e in errors), errors

    def test_error_only_trajectory_not_flagged(self):
        doc = {
            "schema_version": "ATIF-v1.6",
            "session_id": "x",
            "agent": {"name": "openclaw", "version": "1.0.0"},
            "steps": [{"step_id": 1, "source": "system", "message": "exit code 1"}],
        }
        assert validate_atif(doc, strict=True) == []

    def test_empty_agent_turns_not_flagged(self):
        doc = {
            "schema_version": "ATIF-v1.6",
            "session_id": "x",
            "agent": {"name": "openclaw", "version": "1.0.0"},
            "steps": [{"step_id": 1, "source": "agent", "message": ""}],
        }
        assert validate_atif(doc, strict=True) == []


class TestAssertHelper:
    def test_raises_on_invalid(self):
        doc = _good_doc()
        del doc["schema_version"]
        with pytest.raises(ATIFValidationError) as exc:
            assert_atif_compliant(doc)
        assert "missing required field 'schema_version'" in str(exc.value)

    def test_strict_agent_only_raises(self):
        doc = {
            "schema_version": "ATIF-v1.6",
            "session_id": "x",
            "agent": {"name": "a", "version": "1"},
            "steps": [{"step_id": 1, "source": "agent", "message": "done"}],
        }
        with pytest.raises(ATIFValidationError):
            assert_atif_compliant(doc, strict=True)


class TestBuildAtifTrajectoryCompliance:
    def test_empty_steps_list_produces_valid_doc(self):
        traj = build_atif_trajectory([], agent_name="a", agent_version="1")
        errors = validate_atif(traj, strict=False)
        assert any("steps: must be non-empty" in e for e in errors), errors

    def test_minimal_user_turn_strict_compliant(self):
        traj = build_atif_trajectory(
            [{"source": "user", "message": "hi"}, {"source": "agent", "message": "hello"}],
            agent_name="test",
            agent_version="0.0.1",
        )
        assert_atif_compliant(traj, strict=True)

    def test_step_ids_are_auto_numbered(self):
        traj = build_atif_trajectory(
            [{"source": "user", "message": "a"}, {"source": "agent", "message": "b"}],
        )
        assert [s["step_id"] for s in traj[0]["steps"]] == [1, 2]

    def test_step_ids_are_renumbered_even_when_caller_provides_bad_values(self):
        traj = build_atif_trajectory(
            [
                {"source": "user", "message": "a", "step_id": 99},
                {"source": "agent", "message": "b", "step_id": 7},
            ],
        )
        assert [s["step_id"] for s in traj[0]["steps"]] == [1, 2]
        assert_atif_compliant(traj, strict=True)

    def test_user_prompt_kwarg_prepends_user_step(self):
        traj = build_atif_trajectory(
            [{"source": "agent", "message": "I wrote the file."}],
            user_prompt="Create a file called hello.txt",
        )
        steps = traj[0]["steps"]
        assert steps[0] == {
            "source": "user",
            "message": "Create a file called hello.txt",
            "step_id": 1,
        }
        assert_atif_compliant(traj, strict=True)

    def test_user_prompt_noop_when_steps_already_begin_with_user(self):
        traj = build_atif_trajectory(
            [
                {"source": "user", "message": "original"},
                {"source": "agent", "message": "response"},
            ],
            user_prompt="should be ignored",
        )
        assert traj[0]["steps"][0]["message"] == "original"
        assert len(traj[0]["steps"]) == 2

    def test_user_prompt_noop_when_steps_begin_with_system(self):
        traj = build_atif_trajectory(
            [
                {"source": "system", "message": "You are a helpful assistant."},
                {"source": "agent", "message": "ok"},
            ],
            user_prompt="unused",
        )
        assert traj[0]["steps"][0]["source"] == "system"
        assert len(traj[0]["steps"]) == 2

    def test_user_prompt_with_system_prompt_creates_prelude(self):
        traj = build_atif_trajectory(
            [{"source": "agent", "message": "done"}],
            user_prompt="do the thing",
            system_prompt="Be helpful",
        )
        sources = [s["source"] for s in traj[0]["steps"]]
        assert sources == ["system", "user", "agent"]
        assert_atif_compliant(traj, strict=True)


class TestBuildSingleTurnAtifCompliance:
    def test_chat_like_single_turn_strict_compliant(self):
        traj = build_single_turn_atif(
            prompt="What is 2+2?",
            response="4",
            prompt_tokens=3,
            completion_tokens=1,
        )
        assert_atif_compliant(traj, strict=True)
        sources = [s["source"] for s in traj[0]["steps"]]
        assert sources[:2] == ["user", "agent"]

    def test_single_turn_with_system_prelude(self):
        traj = build_single_turn_atif(
            prompt="hi",
            response="hello",
            system="You are a concise assistant.",
        )
        assert [s["source"] for s in traj[0]["steps"]] == ["system", "user", "agent"]
        assert_atif_compliant(traj, strict=True)

    def test_empty_agent_response_still_strict_compliant(self):
        traj = build_single_turn_atif(prompt="write answer.txt", response="")
        assert_atif_compliant(traj, strict=True)


class TestRegressionOpenClawStyleTrajectory:
    AGENT_ONLY_STEPS = [
        {"source": "agent", "message": "Let me check the transcript."},
        {
            "source": "agent",
            "message": "",
            "tool_calls": [{"tool_call_id": "c1", "function_name": "Read", "arguments": {"path": "t.md"}}],
            "observation": {"results": [{"source_call_id": "c1", "content": "file contents"}]},
        },
        {"source": "agent", "message": "I've written the blog post."},
    ]

    def test_pre_fix_agent_only_trajectory_fails_strict(self):
        traj = build_atif_trajectory(copy.deepcopy(self.AGENT_ONLY_STEPS), agent_name="openclaw")
        assert validate_atif(traj, strict=False) == []
        errors = validate_atif(traj, strict=True)
        assert any("no user or system step" in e for e in errors), errors

    def test_post_fix_passes_strict(self):
        traj = build_atif_trajectory(
            copy.deepcopy(self.AGENT_ONLY_STEPS),
            agent_name="openclaw",
            user_prompt="Summarize the meeting transcript into a blog post.",
        )
        assert_atif_compliant(traj, strict=True)

        # The synthesized user step must preserve ATIF sequencing.
        steps = traj[0]["steps"]
        assert [s["step_id"] for s in steps] == list(range(1, len(steps) + 1))
        assert steps[0]["source"] == "user"
        assert steps[0]["message"].startswith("Summarize the meeting")
