import pytest

from nemo_evaluator.models import RetryConfig
from nemo_evaluator.runner.model_client import ModelClient


class TestModelClient:
    def test_strips_chat_completions_from_url(self):
        c = ModelClient(base_url="https://api.example.com/v1/chat/completions", model="test")
        assert c.base_url == "https://api.example.com/v1"

    def test_strips_trailing_slash(self):
        c = ModelClient(base_url="https://api.example.com/v1/", model="test")
        assert c.base_url == "https://api.example.com/v1"

    def test_connection_pooling_reuses_client(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        assert c._get_client() is c._get_client()

    @pytest.mark.asyncio
    async def test_close_releases_client(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        c._get_client()
        await c.close()
        assert c._http is None

    def test_exponential_backoff(self):
        c = ModelClient(
            base_url="https://api.example.com/v1", model="test",
            retry=RetryConfig(base_delay=1.0, max_delay=60.0, jitter=False),
        )
        assert c._backoff_delay(0) == 1.0
        assert c._backoff_delay(1) == 2.0
        assert c._backoff_delay(2) == 4.0
        assert c._backoff_delay(10) == 60.0

    def test_backoff_jitter_stays_within_bounds(self):
        c = ModelClient(
            base_url="https://api.example.com/v1", model="test",
            retry=RetryConfig(base_delay=1.0, jitter=True),
        )
        delays = [c._backoff_delay(0) for _ in range(100)]
        assert min(delays) >= 0.5
        assert max(delays) <= 1.5

    def test_parse_response_extracts_content_and_tokens(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        data = {
            "choices": [{"message": {"content": "42"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            "model": "test",
        }
        resp = c._parse_response(data, 100.0, "What is 6*7?", None)
        assert resp.content == "42"
        assert resp.total_tokens == 15
        assert resp.request_prompt == "What is 6*7?"

    def test_parse_response_rejects_empty_choices(self):
        c = ModelClient(base_url="https://api.example.com/v1", model="test")
        with pytest.raises(ValueError, match="No choices"):
            c._parse_response({"choices": []}, 100.0, "test", None)
