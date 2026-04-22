from copy import deepcopy
from types import SimpleNamespace

import pytest
from nemo_evaluator.sdk.enums import ModelFormat
from nemo_evaluator.sdk.inference import ClientInferenceError
from nemo_evaluator.sdk.metrics.llm_judge import (
    LLMJudgeMetric,
    default_judge_prompt_template_chat,
    default_judge_prompt_template_completions,
    generate_structured_output,
)
from nemo_evaluator.sdk.structured_output import StructuredOutputMode
from nemo_evaluator.sdk.values.common import SecretRef, SupportedJobTypes
from nemo_evaluator.sdk.values.models import Model
from nemo_evaluator.sdk.values.scores import (
    JSONScoreParser,
    RangeScore,
    RegexScoreParser,
    Rubric,
    RubricScore,
    ScoreParserJSON,
)
from pytest_mock import MockerFixture


def _make_metric_score(name: str = "helpfulness") -> RangeScore:
    return RangeScore(
        name=name,
        minimum=1,
        maximum=5,
        parser=JSONScoreParser(json_path=name),
    )


def _make_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/chat/completions",
        name="judge-model",
        format=ModelFormat.OPEN_AI,
    )


def _make_completion_model() -> Model:
    return Model(
        url="https://judge.example.test/v1/completions",
        name="judge-model",
        format=ModelFormat.OPEN_AI,
    )


class TestLLMJudgeMetric:
    def test_rejects_duplicate_score_names(self):
        with pytest.raises(ValueError, match="score names must be unique"):
            LLMJudgeMetric(
                model=_make_model(),
                scores=[_make_metric_score("helpfulness"), _make_metric_score("helpfulness")],
            )

    def test_rejects_reserved_prompt_template_keys(self):
        with pytest.raises(ValueError, match="prompt_template cannot include system_prompt"):
            LLMJudgeMetric(
                model=_make_model(),
                scores=[_make_metric_score()],
                prompt_template={"system_prompt": "bad"},
            )

    def test_offline_default_prompt_template_uses_offline_default(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            job_type=SupportedJobTypes.OFFLINE,
        )

        assert metric.prompt_template == default_judge_prompt_template_chat(SupportedJobTypes.OFFLINE)

    def test_offline_completion_prompt_template_uses_completion_default(self):
        metric = LLMJudgeMetric(
            model=_make_completion_model(),
            scores=[_make_metric_score()],
            job_type=SupportedJobTypes.OFFLINE,
        )

        assert metric.prompt_template == default_judge_prompt_template_completions(SupportedJobTypes.OFFLINE)

    def test_generates_structured_output_from_scores(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )

        assert metric.structured_output == {
            "schema": {
                "type": "object",
                "properties": {
                    "helpfulness": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["helpfulness"],
            }
        }

    def test_initializes_json_parser_with_auto_generated_structured_output(self):
        """Ensure first construction validates JSON parsers against the derived schema."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
        )

        parser = metric._parsers["helpfulness"]

        assert isinstance(parser, ScoreParserJSON)
        assert parser.structured_output == metric.structured_output
        assert parser.json_schema == metric.structured_output["schema"]

    def test_score_names_match_initialized_parsers(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        assert metric.score_names() == ["helpfulness"]

    def test_unique_scores_allows_empty_scores_for_constructed_model(self):
        metric = LLMJudgeMetric.model_construct(model=_make_model(), scores=[], _fields_set={"model", "scores"})
        assert metric.unique_scores() is metric  # type: ignore[call-non-callable]  # Pydantic model_validator descriptor

    def test_string_prompt_template_bypasses_reserved_key_validation(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template="Judge: {{sample.output_text}}",
        )
        assert metric.prompt_template == "Judge: {{sample.output_text}}"

    @pytest.mark.asyncio
    async def test_resolve_secrets_raises_when_secret_is_missing(self):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"api_key_secret": SecretRef(root="judge-secret")}),
            scores=[_make_metric_score()],
        )

        async def resolver(_: str) -> str | None:
            return None

        with pytest.raises(ValueError, match="Missing secret 'judge-secret'"):
            await metric.resolve_secrets(resolver)

    def test_secrets_returns_mapping_when_model_has_api_key_secret(self):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"api_key_secret": SecretRef(root="judge-secret")}),
            scores=[_make_metric_score()],
        )
        assert metric.secrets() == {"judge_secret": SecretRef(root="judge-secret")}

    def test_secrets_returns_empty_mapping_without_api_key_secret(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        assert metric.secrets() == {}

    def test_metric_threshold_score_rejects_unknown_score(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        with pytest.raises(ValueError, match="score name does not match"):
            metric.metric_threshold_score = "missing"

    def test_initialize_score_parsers_requires_parser(self):
        with pytest.raises(ValueError, match="parser is required for LLM-as-a-Judge score helpfulness"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[SimpleNamespace(name="helpfulness", parser=None)],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_initialize_score_parsers_rejects_unknown_parser_type(self):
        with pytest.raises(ValueError, match="unknown parser type for LLM-as-a-Judge score helpfulness"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[SimpleNamespace(name="helpfulness", parser=SimpleNamespace(type="unknown_parser"))],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_handle_none_output_error_mentions_reasoning_only_responses(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        error = metric._handle_none_output_error(
            {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "reasoning_content": "Need more tokens",
                        }
                    }
                ]
            }
        )
        assert "reasoning output but no final text content" in str(error)

    def test_handle_invalid_output_returns_fallback_when_ignore_inference_failure_enabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()], ignore_inference_failure=True)
        assert metric._handle_invalid_output(ValueError("boom"), "fallback", "ignored") == "fallback"

    def test_handle_invalid_output_raises_when_ignore_inference_failure_disabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        with pytest.raises(ValueError, match="boom"):
            metric._handle_invalid_output(ValueError("boom"), "fallback", "ignored")

    def test_validate_output_text_raises_for_none_output(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        with pytest.raises(ValueError, match="LLM judge returned no usable textual content"):
            metric._validate_output_text(None, {"choices": [{"message": {"content": None}}]})

    def test_render_request_warns_on_overlapping_keys_and_includes_score_dumps(self, mocker: MockerFixture):
        warn = mocker.patch("nemo_evaluator.sdk.metrics.llm_judge._logger.warning")
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            prompt_template={"messages": [{"role": "user", "content": "{{prompt}} {{scores.helpfulness.minimum}}"}]},
        )

        request = metric._render_request({"prompt": "item"}, {"prompt": "sample"})

        warn.assert_called_once()
        assert request["messages"][0]["content"] == "sample 1"
        assert request["max_tokens"] == 1024

    def test_retry_with_max_completion_tokens_updates_request(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        request = {"max_tokens": 64, "messages": []}

        rewritten = metric._retry_with_max_completion_tokens(request)

        assert metric._use_max_completion_tokens is True
        assert rewritten == {"max_completion_tokens": 64, "messages": []}

    def test_render_request_uses_max_completion_tokens_when_flag_enabled(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        metric._use_max_completion_tokens = True

        request = metric._render_request({"prompt": "hello"}, {"output_text": "world"})

        assert "max_tokens" not in request
        assert request["max_completion_tokens"] == 1024

    def test_metric_requires_threshold_configuration(self):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        with pytest.raises(ValueError, match="metric_threshold and metric_threshold_score are required"):
            metric.metric({"prompt": "hello"}, {"output_text": "world"})

    def test_metric_retries_with_max_completion_tokens(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            metric_threshold=2.5,
        )
        metric.metric_threshold_score = "helpfulness"
        error = ClientInferenceError(
            mocker.Mock(status_code=400, response=mocker.Mock(text="'max_tokens' is not supported with this model"))
        )
        captured_requests: list[dict] = []

        async def inference_fn(*args, **kwargs):
            request = kwargs.get("request", args[1])
            captured_requests.append(deepcopy(request))
            if len(captured_requests) == 1:
                raise error
            return {"choices": [{"message": {"content": '{"helpfulness": 3}'}}]}

        metric.set_inference_fn(inference_fn)

        assert metric.metric({"prompt": "hello"}, {"output_text": "world"}, trace=True) is True
        assert len(captured_requests) == 2
        first_request = captured_requests[0]
        second_request = captured_requests[1]
        assert "max_tokens" in first_request
        assert "max_completion_tokens" in second_request

    def test_metric_returns_score_when_trace_is_disabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            metric_threshold=2.5,
        )
        metric.metric_threshold_score = "helpfulness"
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": '{"helpfulness": 3}'}}]})
        )

        assert metric.metric({"prompt": "hello"}, {"output_text": "world"}) == 3

    def test_metric_invalid_output_returns_false_for_trace_when_ignore_enabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            metric_threshold=2.5,
            ignore_inference_failure=True,
        )
        metric.metric_threshold_score = "helpfulness"
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": None, "reasoning_content": None}}]})
        )

        assert metric.metric({"prompt": "hello"}, {"output_text": "world"}, trace=True) is False

    def test_metric_invalid_output_returns_nan_when_trace_is_disabled_and_ignore_enabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            metric_threshold=2.5,
            ignore_inference_failure=True,
        )
        metric.metric_threshold_score = "helpfulness"
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": None, "reasoning_content": None}}]})
        )

        score = metric.metric({"prompt": "hello"}, {"output_text": "world"}, trace=False)
        assert score != score

    @pytest.mark.asyncio
    async def test_compute_scores_returns_nan_when_inference_failure_is_ignored(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            ignore_inference_failure=True,
        )
        error = ClientInferenceError(mocker.Mock(status_code=500, response=mocker.Mock(text="boom")))
        metric.set_inference_fn(mocker.AsyncMock(side_effect=error))

        result = await metric.compute_scores({"prompt": "hello"}, {"output_text": "world"})

        assert len(result.scores) == 1
        assert result.scores[0].name == "helpfulness"
        assert result.scores[0].value != result.scores[0].value

    def test_metric_returns_nan_when_inference_failure_is_ignored(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            metric_threshold=2.5,
            ignore_inference_failure=True,
        )
        metric.metric_threshold_score = "helpfulness"
        error = ClientInferenceError(mocker.Mock(status_code=500, response=mocker.Mock(text="boom")))
        metric.set_inference_fn(mocker.AsyncMock(side_effect=error))

        score = metric.metric({"prompt": "hello"}, {"output_text": "world"})

        assert score != score

    @pytest.mark.asyncio
    async def test_compute_scores_retries_with_max_completion_tokens(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        error = ClientInferenceError(
            mocker.Mock(status_code=400, response=mocker.Mock(text="'max_tokens' is not supported with this model"))
        )
        captured_requests: list[dict] = []

        async def inference_fn(*args, **kwargs):
            request = kwargs.get("request", args[1])
            captured_requests.append(deepcopy(request))
            if len(captured_requests) == 1:
                raise error
            return {"choices": [{"message": {"content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(inference_fn)

        result = await metric.compute_scores({"prompt": "hello"}, {"output_text": "world"})

        assert result.scores[0].value == 4
        assert "max_tokens" in captured_requests[0]
        assert "max_completion_tokens" in captured_requests[1]

    @pytest.mark.asyncio
    async def test_compute_scores_raises_invalid_output_when_ignore_failure_disabled(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(model=_make_model(), scores=[_make_metric_score()])
        metric.set_inference_fn(
            mocker.AsyncMock(return_value={"choices": [{"message": {"content": None, "reasoning_content": None}}]})
        )

        with pytest.raises(ValueError, match="LLM judge returned no usable textual content"):
            await metric.compute_scores({"prompt": "hello"}, {"output_text": "world"})

    @pytest.mark.asyncio
    async def test_preflight_selects_structured_output_mode_for_nim(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model()
            .with_default_headers({"X-NMP-Principal-Id": "service:evaluator"})
            .model_copy(update={"format": ModelFormat.NVIDIA_NIM}),
            scores=[_make_metric_score()],
        )
        detect = mocker.patch(
            "nemo_evaluator.sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
            return_value=StructuredOutputMode.ROOT_GUIDED_JSON,
        )

        await metric.preflight()

        detect.assert_awaited_once()
        assert detect.await_args.kwargs["model"].default_headers == {"X-NMP-Principal-Id": "service:evaluator"}
        structured_hook = next(hook for hook in metric._preprocess_hooks if hasattr(hook, "mode"))
        assert structured_hook.mode == StructuredOutputMode.ROOT_GUIDED_JSON

    @pytest.mark.asyncio
    async def test_preflight_is_noop_without_structured_output(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=RegexScoreParser(pattern="(\\d+)"))],
        )
        detect = mocker.patch(
            "nemo_evaluator.sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
        )

        await metric.preflight()

        detect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_preflight_is_noop_when_structured_hook_is_missing(self, mocker: MockerFixture):
        metric = LLMJudgeMetric(
            model=_make_model().model_copy(update={"format": ModelFormat.NVIDIA_NIM}),
            scores=[_make_metric_score()],
        )
        metric._preprocess_hooks = []
        detect = mocker.patch(
            "nemo_evaluator.sdk.metrics.llm_judge.detect_structured_output_mode",
            new_callable=mocker.AsyncMock,
        )

        await metric.preflight()

        detect.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_compute_scores_passes_model_default_headers_to_inference_fn(self):
        metric = LLMJudgeMetric(
            model=_make_model().with_default_headers({"X-NMP-Principal-Id": "service:evaluator"}),
            scores=[_make_metric_score()],
        )
        captured: dict = {}

        async def inference_fn(
            model: Model,
            request: dict,
            max_retries: int | None,
            api_key: str | None = None,
            *,
            default_headers: dict | None = None,
            timeout: float | None = None,
        ) -> dict:
            del request, max_retries, api_key, default_headers, timeout
            captured["default_headers"] = model.default_headers
            return {"choices": [{"message": {"content": '{"helpfulness": 4}'}}]}

        metric.set_inference_fn(inference_fn)

        result = await metric.compute_scores({"prompt": "hello"}, {"output_text": "world"})

        assert result.scores[0].value == 4
        assert captured["default_headers"] == {"X-NMP-Principal-Id": "service:evaluator"}


class TestGenerateStructuredOutput:
    def test_returns_explicit_structured_output_when_provided(self):
        explicit = {"schema": {"type": "object", "properties": {"helpfulness": {"type": "integer"}}}}
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[_make_metric_score()],
            structured_output=explicit,
        )
        assert generate_structured_output(metric) == explicit

    def test_returns_none_when_no_json_parsed_scores_exist(self):
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="helpfulness", minimum=1, maximum=5, parser=RegexScoreParser(pattern="(\\d+)"))],
        )
        assert generate_structured_output(metric) is None

    def test_raises_for_conflicting_auto_generated_schemas(self):
        with pytest.raises(ValueError, match="conflicting auto-generated structured_output"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[
                    RangeScore(name="helpfulness", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                    RubricScore(
                        name="quality",
                        rubric=[Rubric(label="good", value=1), Rubric(label="bad", value=0)],
                        parser=JSONScoreParser(json_path="score"),
                    ),
                ],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_skips_json_scores_that_are_not_range_or_rubric(self):
        params = SimpleNamespace(
            structured_output=None,
            scores=[SimpleNamespace(name="custom", parser=JSONScoreParser(json_path="custom"))],
        )

        assert generate_structured_output(params) is None

    def test_roundtrip_uses_parser_json_path_for_auto_generated_schema(self):
        """Ensure direct construction and roundtrip reconstruction agree on parser json_path."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[
                RangeScore(
                    name="accuracy",
                    minimum=1,
                    maximum=5,
                    parser=JSONScoreParser(json_path="score"),
                )
            ],
        )

        assert metric.structured_output == {
            "schema": {
                "type": "object",
                "properties": {
                    "score": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 5,
                    }
                },
                "required": ["score"],
            }
        }

        roundtrip_metric = LLMJudgeMetric.model_validate(metric.model_dump(mode="json"))
        parser = roundtrip_metric._parsers["accuracy"]

        assert isinstance(parser, ScoreParserJSON)
        assert parser.json_path == "score"
        assert parser.structured_output == roundtrip_metric.structured_output

    def test_uses_json_path_as_property_key(self):
        """Verify the generated schema keys on json_path, not the score name."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score"))],
        )
        assert generate_structured_output(metric) == {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }

    def test_deduplicates_shared_json_path(self):
        """Two scores sharing the same json_path and identical schema produce a single property."""
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[
                RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                RangeScore(name="relevance", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
            ],
        )
        assert generate_structured_output(metric) == {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }

    def test_rejects_conflicting_ranges_on_shared_json_path(self):
        """Two RangeScores sharing a json_path but with different ranges are rejected."""
        with pytest.raises(ValueError, match="conflicting auto-generated structured_output for json_path 'score'"):
            LLMJudgeMetric.model_construct(
                model=_make_model(),
                scores=[
                    RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score")),
                    RangeScore(name="relevance", minimum=0, maximum=10, parser=JSONScoreParser(json_path="score")),
                ],
                prompt_template=default_judge_prompt_template_chat(),
                structured_output=None,
                _fields_set={"model", "scores", "prompt_template", "structured_output"},
            )

    def test_accepts_preexisting_structured_output_with_custom_json_path(self):
        """Explicit structured_output is preserved even when scores use a custom json_path."""
        explicit = {
            "schema": {
                "type": "object",
                "properties": {"score": {"type": "integer", "minimum": 1, "maximum": 5}},
                "required": ["score"],
            }
        }
        metric = LLMJudgeMetric(
            model=_make_model(),
            scores=[RangeScore(name="accuracy", minimum=1, maximum=5, parser=JSONScoreParser(json_path="score"))],
            structured_output=explicit,
        )
        assert metric.structured_output == explicit
        assert "accuracy" in metric._parsers
