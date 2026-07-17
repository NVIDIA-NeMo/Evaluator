# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import inspect

import pytest

from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.failure_classification import MODEL_FAILURE_CATEGORIES, classify_model_failure
from nemo_evaluator.observability.types import ModelResponse, StepRecord


class TestModelResponse:
    def test_request_hash_depends_on_prompt_not_content(self):
        r1 = ModelResponse(content="42", request_prompt="What is 6*7?", model="test")
        r2 = ModelResponse(content="99", request_prompt="What is 6*7?", model="test")
        r3 = ModelResponse(content="42", request_prompt="What is 7*8?", model="test")
        assert r1.request_hash == r2.request_hash
        assert r1.request_hash != r3.request_hash

    def test_response_hash_depends_on_content_not_prompt(self):
        r1 = ModelResponse(content="42", request_prompt="a")
        r2 = ModelResponse(content="42", request_prompt="b")
        r3 = ModelResponse(content="43", request_prompt="a")
        assert r1.response_hash == r2.response_hash
        assert r1.response_hash != r3.response_hash


class TestStepRecord:
    def test_to_dict_includes_model_hashes(self):
        resp = ModelResponse(content="42", model="test", total_tokens=10, request_prompt="Q", request_system=None)
        step = StepRecord(model_response=resp)
        d = step.to_dict()
        assert d["model"]["content"] == "42"
        assert "request_hash" in d["model"]
        assert "response_hash" in d["model"]

    def test_to_dict_records_error(self):
        step = StepRecord(model_error="Timeout")
        assert step.to_dict()["model_error"] == "Timeout"


class TestArtifactCollector:
    @pytest.mark.parametrize(
        ("error_msg", "expected_category"),
        [
            ("Request timed out after 120s", "model_timeout"),
            ("429 Too Many Requests", "rate_limit"),
            ("503 Service Unavailable", "server_error"),
            ("Error code: 400 - Malformed native tool-call JSON", "model_error"),
            ("APIStatusError: status_code=418", "model_error"),
            ("Turn budget exhausted: 20/20 turns used", "turn_budget_exhausted"),
            (
                "litellm.RateLimitError: Turn budget exhausted: 215/200 turns used. session 6a83baf1-cc335504",
                "turn_budget_exhausted",
            ),
            ("litellm.ContextWindowExceededError: maximum context length is 262144 tokens", "context_window_exceeded"),
            (
                "litellm.BudgetExceededError: Budget has been exceeded! Current cost: 2, Max budget: 1",
                "budget_exceeded",
            ),
            ("litellm.RateLimitError: Too many requests", "rate_limit"),
            ("litellm.RouterRateLimitError: router cooldown active", "rate_limit"),
            ("litellm.RouterRateLimitErrorBasic: router cooldown active", "rate_limit"),
            ("litellm.RateLimitType: requests per minute", "rate_limit"),
            ("litellm.APITimeoutError: Request timed out", "model_timeout"),
            ("litellm.Timeout: request timed out", "model_timeout"),
            ("litellm.BadGatewayError: MidStreamFallbackError: APIConnectionError", "server_error"),
            ("litellm.MidStreamFallbackError: No fallback model group found", "server_error"),
            ("litellm.APIConnectionError: connection reset by peer", "server_error"),
            ("litellm.APIError: provider returned an unexpected server response", "server_error"),
            ("litellm.ServiceUnavailableError: service unavailable", "server_error"),
            ("litellm.InternalServerError: internal server error", "server_error"),
            ("ServerDisconnectedError: upstream closed", "server_error"),
            ("litellm.APIResponseValidationError: response was missing choices", "model_error"),
            ("litellm.JSONSchemaValidationError: returned an invalid response", "model_error"),
            ("litellm.AuthenticationError: invalid API key", "model_error"),
            ("litellm.PermissionDeniedError: permission denied", "model_error"),
            ("litellm.NotFoundError: model not found", "model_error"),
            ("litellm.UnprocessableEntityError: invalid payload", "model_error"),
            ("litellm.UnsupportedParamsError: unsupported response_format", "model_error"),
            ("litellm.ContentPolicyViolationError: blocked by content policy", "model_error"),
            ("litellm.ImageFetchError: could not fetch input image", "model_error"),
            ("litellm.LiteLLMUnknownProvider: unknown model provider", "model_error"),
            ("litellm.MockException: synthetic model client failure", "model_error"),
            ("litellm.InvalidRequestError: bad request", "model_error"),
            ("litellm.RejectedRequestError: request rejected", "model_error"),
            ("litellm.BlockedPiiEntityError: blocked PII entity", "model_error"),
            ("litellm.GuardrailInterventionNormalStringError: guardrail intervention", "model_error"),
            ("litellm.GuardrailRaisedException: guardrail raised", "model_error"),
            ("litellm.ErrorEventError: stream emitted error event", "model_error"),
            ("litellm.ModifyResponseException: response modifier failed", "model_error"),
            ("litellm.SensitiveDataRouteException: sensitive data route rejected", "model_error"),
            ("litellm.OpenAIError: untyped model client failure", "model_error"),
        ],
    )
    def test_classify_model_failure(self, error_msg, expected_category):
        assert classify_model_failure(error_msg) == expected_category

    def test_all_installed_litellm_exception_names_classify(self):
        import litellm.exceptions as litellm_exceptions

        exception_names = sorted(
            name
            for name, obj in vars(litellm_exceptions).items()
            if inspect.isclass(obj) and obj.__module__ == litellm_exceptions.__name__
        )

        assert exception_names
        missing = [
            name for name in exception_names if classify_model_failure(f"litellm.{name}: synthetic failure") is None
        ]
        assert missing == []

    def test_status_codes_are_not_matched_inside_identifiers(self):
        error = "Agent crashed: ConversationRunError: conversation 6a83baf1-cc335504 had no status code"

        assert classify_model_failure(error) is None

    @pytest.mark.parametrize(
        ("error", "status_code"),
        [
            pytest.param("Gateway Timeout", 504, id="explicit-status"),
            pytest.param("HTTP 504 Gateway Timeout", None, id="http"),
            pytest.param("504 Gateway Time-out", None, id="leading-status"),
            pytest.param("HTTP/1.1 504 Gateway Time-out", None, id="http-version"),
        ],
    )
    def test_gateway_timeout_is_a_server_error(self, error, status_code):
        assert classify_model_failure(error, status_code=status_code) == "server_error"

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param("upstream timeout", id="upstream-timeout"),
            pytest.param("upstream timed out", id="upstream-timed-out"),
            pytest.param("upstream timeout: provider stalled", id="upstream-timeout-prefix"),
            pytest.param("gateway timeout: upstream stalled", id="gateway-timeout-prefix"),
            pytest.param("gateway timeout error", id="gateway-timeout-error"),
            pytest.param("upstream timeout error", id="upstream-timeout-error"),
        ],
    )
    def test_server_timeout_phrases_are_server_errors(self, error):
        assert classify_model_failure(error) == "server_error"

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param("429 model calls completed successfully", id="model-call-count"),
            pytest.param("500 samples evaluated successfully", id="sample-count"),
        ],
    )
    def test_leading_counts_are_not_status_codes(self, error):
        assert classify_model_failure(error) is None

    @pytest.mark.parametrize(
        ("error", "expected_category"),
        [
            pytest.param("rate limit exceeded", "rate_limit", id="spaced-rate-limit"),
            pytest.param(
                "litellm.BadRequestError: You passed 171337 input tokens and requested 0 output tokens. "
                "However, the model's context length is only 163840 tokens. ...",
                "context_window_exceeded",
                id="vllm-context-length",
            ),
        ],
    )
    def test_provider_failure_phrases_are_classified(self, error, expected_category):
        assert classify_model_failure(error) == expected_category

    @pytest.mark.parametrize(
        "status_code",
        [200, 201, 204, 302],
    )
    def test_success_statuses_are_not_model_failures(self, status_code):
        assert classify_model_failure(status_code=status_code) is None
        assert classify_model_failure(f"HTTP {status_code}") is None

    def test_failure_classification(self):
        c = ArtifactCollector()
        cases = [
            ("Request timed out after 120s", "model_timeout"),
            ("429 Too Many Requests", "rate_limit"),
            ("503 Service Unavailable", "server_error"),
            ("Error code: 400 - Malformed native tool-call JSON", "model_error"),
            ("Turn budget exhausted: 20/20 turns used", "turn_budget_exhausted"),
        ]
        for error_msg, expected_category in cases:
            step = StepRecord(model_error=error_msg)
            c.record(step)
            assert step.failure_category == expected_category, f"Expected {expected_category} for: {error_msg}"

    def test_unclassified_model_error_falls_back_to_model_error(self):
        c = ArtifactCollector()
        c.record(StepRecord(model_error="Agent crashed: ValueError"))

        assert c.build(elapsed=0.0).steps[0].failure_category == "model_error"

    @pytest.mark.parametrize(
        "error",
        [
            "500 Server Error for http+docker://localhost/v1.41/containers/create",
            "container exited with status 137",
            "Verifier worker stopped unexpectedly",
        ],
        ids=["docker-daemon-500", "container-exit-137", "unclassified-verifier"],
    )
    def test_system_errors_remain_non_model_failures(self, error):
        collector = ArtifactCollector()
        step = StepRecord(
            scoring_details={
                "error": error,
                "error_category": "system",
                "method": "system_error",
                "retries_exhausted": 2,
            }
        )

        collector.record(step)
        report = collector.build(elapsed=1.0).failures

        assert step.failure_category == "system"
        assert step.failure_category not in MODEL_FAILURE_CATEGORIES
        assert set(report.categories) == {"system"}

    def test_graceful_turn_budget_error_remains_model_classified(self):
        collector = ArtifactCollector()
        step = StepRecord(
            scoring_details={
                "error": "Turn budget exhausted: 20/20 turns used",
                "error_category": "graceful",
            }
        )

        collector.record(step)

        assert step.failure_category == "turn_budget_exhausted"

    @pytest.mark.parametrize(
        ("scoring_details", "expected_category"),
        [
            (
                {"error": "Error code: 400 - Malformed native tool-call JSON", "error_category": "graceful"},
                "model_error",
            ),
            ({"error": "504 Gateway Timeout", "error_category": "model_timeout"}, "model_timeout"),
        ],
    )
    def test_scoring_error_classified_even_when_content_empty(self, scoring_details, expected_category):
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="  "), scoring_details=scoring_details)
        c.record(step)
        assert step.failure_category == expected_category

    def test_empty_content_yields_no_failure_category(self):
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="  "))

        c.record(step)
        artifacts = c.build(elapsed=1.0)

        assert step.failure_category is None
        assert artifacts.failures.total_failures == 0

    def test_empty_content_with_zero_reward_is_not_format_error(self):
        # Regression guard (FEP-1132): empty content must terminate
        # classification, not fall through to the reward==0 +
        # extracted_answer-is-None format_error check.
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="  "), reward=0.0)
        c.record(step)
        assert step.failure_category is None

    def test_refusal_detected(self):
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="I cannot help with that"))
        c.record(step)
        assert step.failure_category == "refusal"

    def test_custom_refusal_patterns(self):
        c = ArtifactCollector(refusal_patterns=[r"NOPE"])
        step = StepRecord(model_response=ModelResponse(content="NOPE, won't do it"))
        c.record(step)
        assert step.failure_category == "refusal"

    def test_runtime_aggregation(self):
        c = ArtifactCollector()
        for _ in range(10):
            c.record(
                StepRecord(
                    model_response=ModelResponse(
                        content="x", total_tokens=100, prompt_tokens=50, completion_tokens=50, latency_ms=100.0
                    ),
                    model_ms=100.0,
                )
            )
        arts = c.build(elapsed=10.0)
        assert arts.runtime.total_steps == 10
        assert arts.runtime.total_tokens == 1000
        assert arts.runtime.tokens_per_second == 100.0

    def test_retries_summed(self):
        c = ArtifactCollector()
        c.record(StepRecord(retries=2, model_response=ModelResponse(content="ok")))
        c.record(StepRecord(retries=1, model_response=ModelResponse(content="ok")))
        assert c.build(1.0).runtime.total_retries == 3
