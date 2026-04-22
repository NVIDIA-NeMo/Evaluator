import pytest
from nemo_evaluator.sdk.inference import (
    _openai_clients,
    make_inference_request,
    merge_default_headers,
    redact_request_for_logging,
    requests_log_var,
)
from nemo_evaluator.sdk.values.models import Model
from pytest_mock import MockerFixture


class TestInferenceHelpers:
    def test_merge_default_headers_combines_model_and_request_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-Trace-Id": "model", "X-Model": "judge"},
        )

        assert merge_default_headers(model, {"X-Trace-Id": "request", "X-Request": "abc"}) == {
            "X-Trace-Id": "request",
            "X-Model": "judge",
            "X-Request": "abc",
        }

    def test_merge_default_headers_returns_none_when_both_sources_are_empty(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
        )

        assert merge_default_headers(model, None) is None

    def test_redact_request_for_logging_filters_only_auth_headers(self):
        assert redact_request_for_logging(
            {
                "model": "judge-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {
                    "Authorization": "Bearer secret-token",
                    "X-Trace-Id": "trace-123",
                },
            }
        ) == {
            "model": "judge-model",
            "messages": [{"role": "user", "content": "hi"}],
            "extra_headers": {"X-Trace-Id": "trace-123"},
        }

    def test_redact_request_for_logging_omits_extra_headers_when_all_are_filtered(self):
        assert redact_request_for_logging(
            {
                "model": "judge-model",
                "messages": [{"role": "user", "content": "hi"}],
                "extra_headers": {
                    "Authorization": "Bearer secret-token",
                    "x-auth-token": "secret-token",
                },
            }
        ) == {
            "model": "judge-model",
            "messages": [{"role": "user", "content": "hi"}],
        }


class TestMakeInferenceRequest:
    @pytest.fixture(autouse=True)
    def clear_client_cache(self):
        _openai_clients.clear()
        yield
        _openai_clients.clear()

    @staticmethod
    def _make_model(**kwargs) -> Model:
        return Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_uses_model_default_headers_as_request_extra_headers(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-NMP-Principal-Id": "service:evaluator"})

        mock_openai = mocker.patch("nemo_evaluator.sdk.inference.AsyncOpenAI")
        mock_instance = mock_openai.return_value
        mock_with_opts = mocker.MagicMock()
        mock_instance.with_options.return_value = mock_with_opts
        mock_with_opts.chat = mocker.MagicMock()
        mock_with_opts.chat.completions.create = mocker.AsyncMock(
            return_value=mocker.MagicMock(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}]})
        )

        await make_inference_request(model, {"messages": [{"role": "user", "content": "hi"}]})

        request_body = mock_with_opts.chat.completions.create.call_args.kwargs
        assert request_body["extra_headers"] == {"X-NMP-Principal-Id": "service:evaluator"}

    @pytest.mark.asyncio
    async def test_redacts_extra_headers_from_persisted_request_log(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-Trace-Id": "trace-123"})

        mock_openai = mocker.patch("nemo_evaluator.sdk.inference.AsyncOpenAI")
        mock_instance = mock_openai.return_value
        mock_with_opts = mocker.MagicMock()
        mock_instance.with_options.return_value = mock_with_opts
        mock_with_opts.chat = mocker.MagicMock()
        mock_with_opts.chat.completions.create = mocker.AsyncMock(
            return_value=mocker.MagicMock(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}]})
        )

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"Authorization": "Bearer secret-token"},
        )

        request_body = mock_with_opts.chat.completions.create.call_args.kwargs
        assert request_body["extra_headers"] == {
            "X-Trace-Id": "trace-123",
            "Authorization": "Bearer secret-token",
        }
        assert request_log[0]["request"]["extra_headers"] == {"X-Trace-Id": "trace-123"}

    @pytest.mark.asyncio
    async def test_filters_cookie_headers_from_persisted_request_log(self, mocker: MockerFixture):
        model = self._make_model()

        mock_openai = mocker.patch("nemo_evaluator.sdk.inference.AsyncOpenAI")
        mock_instance = mock_openai.return_value
        mock_with_opts = mocker.MagicMock()
        mock_instance.with_options.return_value = mock_with_opts
        mock_with_opts.chat = mocker.MagicMock()
        mock_with_opts.chat.completions.create = mocker.AsyncMock(
            return_value=mocker.MagicMock(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}]})
        )

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"Cookie": "session=abc", "X-Trace-Id": "trace-123"},
        )

        request_body = mock_with_opts.chat.completions.create.call_args.kwargs
        assert request_body["extra_headers"] == {
            "Cookie": "session=abc",
            "X-Trace-Id": "trace-123",
        }
        assert request_log[0]["request"]["extra_headers"] == {"X-Trace-Id": "trace-123"}

    @pytest.mark.asyncio
    async def test_omits_logged_extra_headers_when_all_headers_are_auth_style(self, mocker: MockerFixture):
        model = self._make_model()

        mock_openai = mocker.patch("nemo_evaluator.sdk.inference.AsyncOpenAI")
        mock_instance = mock_openai.return_value
        mock_with_opts = mocker.MagicMock()
        mock_instance.with_options.return_value = mock_with_opts
        mock_with_opts.chat = mocker.MagicMock()
        mock_with_opts.chat.completions.create = mocker.AsyncMock(
            return_value=mocker.MagicMock(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}]})
        )

        request_log: list[dict] = []
        requests_log_var.set(request_log)

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={
                "Authorization": "Bearer secret-token",
                "x-auth-token": "secret-token",
                "Cookie": "session=abc",
                "Set-Cookie": "session=def",
            },
        )

        request_body = mock_with_opts.chat.completions.create.call_args.kwargs
        assert request_body["extra_headers"] == {
            "Authorization": "Bearer secret-token",
            "x-auth-token": "secret-token",
            "Cookie": "session=abc",
            "Set-Cookie": "session=def",
        }
        assert "extra_headers" not in request_log[0]["request"]

    @pytest.mark.asyncio
    async def test_per_call_headers_override_model_default_headers(self, mocker: MockerFixture):
        model = self._make_model(default_headers={"X-Trace-Id": "model", "X-Model": "judge"})

        mock_openai = mocker.patch("nemo_evaluator.sdk.inference.AsyncOpenAI")
        mock_instance = mock_openai.return_value
        mock_with_opts = mocker.MagicMock()
        mock_instance.with_options.return_value = mock_with_opts
        mock_with_opts.chat = mocker.MagicMock()
        mock_with_opts.chat.completions.create = mocker.AsyncMock(
            return_value=mocker.MagicMock(model_dump=lambda: {"choices": [{"message": {"content": "ok"}}]})
        )

        await make_inference_request(
            model,
            {"messages": [{"role": "user", "content": "hi"}]},
            default_headers={"X-Trace-Id": "request", "X-Request": "abc"},
        )

        request_body = mock_with_opts.chat.completions.create.call_args.kwargs
        assert request_body["extra_headers"] == {
            "X-Trace-Id": "request",
            "X-Model": "judge",
            "X-Request": "abc",
        }
