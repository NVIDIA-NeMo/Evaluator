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
import pytest

from nemo_evaluator.observability.collector import ArtifactCollector
from nemo_evaluator.observability.failure_classification import classify_model_failure
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
            ("litellm.APITimeoutError: Request timed out", "model_timeout"),
            ("litellm.Timeout: request timed out", "model_timeout"),
            ("litellm.BadGatewayError: MidStreamFallbackError: APIConnectionError", "server_error"),
            ("litellm.MidStreamFallbackError: No fallback model group found", "server_error"),
            ("litellm.APIConnectionError: connection reset by peer", "server_error"),
            ("litellm.APIError: provider returned an unexpected server response", "server_error"),
            ("litellm.ServiceUnavailableError: service unavailable", "server_error"),
            ("litellm.InternalServerError: internal server error", "server_error"),
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
            ("litellm.OpenAIError: untyped model client failure", "model_error"),
        ],
    )
    def test_classify_model_failure(self, error_msg, expected_category):
        assert classify_model_failure(error_msg) == expected_category

    def test_status_codes_are_not_matched_inside_identifiers(self):
        error = "Agent crashed: ConversationRunError: conversation 6a83baf1-cc335504 had no status code"

        assert classify_model_failure(error) is None

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
    def test_scoring_error_preempts_empty_response(self, scoring_details, expected_category):
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="  "), scoring_details=scoring_details)
        c.record(step)
        assert step.failure_category == expected_category

    def test_empty_response_detected(self):
        c = ArtifactCollector()
        step = StepRecord(model_response=ModelResponse(content="  "))
        c.record(step)
        assert step.failure_category == "empty_response"

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
