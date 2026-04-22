import pytest
from nemo_evaluator.sdk.values.models import (
    Model,
    filter_auth_headers,
    is_auth_header_name,
    normalize_header_name,
)
from pydantic_core import ValidationError


class TestHeaderNameHelpers:
    def test_normalize_header_name_strips_lowercases_and_replaces_underscores(self):
        assert normalize_header_name("  X_AUTH_Token  ") == "x-auth-token"

    @pytest.mark.parametrize(
        ("header_name", "expected"),
        [
            ("Authorization", True),
            ("Cookie", True),
            ("Set-Cookie", True),
            ("x-auth-token", True),
            ("openai-api-key", True),
            ("my_secret_header", True),
            ("X-Trace-Id", False),
            ("X-NMP-Principal-Id", False),
        ],
    )
    def test_is_auth_header_name(self, header_name: str, expected: bool):
        assert is_auth_header_name(header_name) is expected

    def test_filter_auth_headers_removes_only_auth_style_headers(self):
        assert filter_auth_headers(
            {
                "Authorization": "Bearer secret-token",
                "X-Trace-Id": "trace-123",
                "X-NMP-Principal-Id": "service:evaluator",
            }
        ) == {
            "X-Trace-Id": "trace-123",
            "X-NMP-Principal-Id": "service:evaluator",
        }

    def test_filter_auth_headers_returns_none_when_all_headers_are_filtered(self):
        assert filter_auth_headers({"Authorization": "Bearer secret-token", "x-auth-token": "secret-token"}) is None


class TestModelDefaultHeaders:
    def test_with_default_headers_returns_copy_with_merged_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-Existing": "model"},
        )

        updated = model.with_default_headers({"X-NMP-Principal-Id": "service:evaluator"})

        assert updated is not model
        assert model.default_headers == {"X-Existing": "model"}
        assert updated.default_headers == {
            "X-Existing": "model",
            "X-NMP-Principal-Id": "service:evaluator",
        }

    def test_model_dump_excludes_default_headers(self):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={"X-NMP-Principal-Id": "service:evaluator"},
        )

        assert "default_headers" not in model.model_dump(mode="python")

    @pytest.mark.parametrize(
        "header_name",
        [
            "Authorization",
            "Cookie",
            "Set-Cookie",
            "Proxy-Authorization",
            "X-API-Key",
            "x-auth-token",
            "openai-api-key",
            "my_secret_header",
        ],
    )
    def test_rejects_auth_style_default_headers(self, header_name: str):
        with pytest.raises(
            ValidationError,
            match="default_headers cannot include authentication headers .*model.api_key_secret",
        ):
            Model(
                url="https://judge.example.test/v1/chat/completions",
                name="judge-model",
                default_headers={header_name: "blocked"},
            )

    @pytest.mark.parametrize(
        "header_name",
        [
            "X-NMP-Principal-Id",
            "X-Trace-Id",
        ],
    )
    def test_allows_non_auth_default_headers(self, header_name: str):
        model = Model(
            url="https://judge.example.test/v1/chat/completions",
            name="judge-model",
            default_headers={header_name: "value"},
        )

        assert model.default_headers == {header_name: "value"}
