"""Model-related value types."""

from __future__ import annotations

import logging
import os
from functools import cached_property
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.config import JsonDict

from nemo_evaluator.sdk.enums import ModelFormat
from nemo_evaluator.sdk.values.common import SecretRef

logger = logging.getLogger(__name__)

_AUTH_HEADER_PATTERNS: tuple[str, ...] = (
    "auth",
    "token",
    "key",
    "secret",
    "credential",
    "bearer",
    "cookie",
    "set-cookie",
)


class ReasoningParams(BaseModel):
    """Custom settings that control the model's reasoning behavior."""

    end_token: str | None = Field(
        default=None,
        description="Configure the end token to trim reasoning context based on the model's reasoning API. Example for Nemotron models: '</think>'",
    )
    include_if_not_finished: bool | None = Field(
        default=None,
        description="Configure whether to include reasoning context if the model has not finished reasoning.",
    )
    effort: str | None = Field(
        default=None,
        description="Option for OpenAI models to specify low, medium, or high reasoning effort.",
    )


def _strip_internal_fields(schema: JsonDict) -> None:
    """Remove internal-only fields from the JSON schema so they don't appear in the OpenAPI spec."""
    props = schema.get("properties")
    if isinstance(props, dict):
        props.pop("default_headers", None)
        props.pop("host_url", None)


def normalize_header_name(header_name: str) -> str:
    """Normalize a transport header name for policy checks."""
    return header_name.strip().lower().replace("_", "-")


def is_auth_header_name(header_name: str) -> bool:
    """Return whether a header name appears to carry authentication material."""
    normalized_header_name = normalize_header_name(header_name)
    return any(pattern in normalized_header_name for pattern in _AUTH_HEADER_PATTERNS)


def filter_auth_headers(headers: dict[str, str] | None) -> dict[str, str] | None:
    """Return only non-auth headers from a header mapping."""
    if headers is None:
        return None

    filtered_headers: dict[str, str] = {}
    for header_name, header_value in headers.items():
        if is_auth_header_name(header_name):
            logger.debug(
                f"Filtered header {header_name} because it was recognized as an auth header",
            )
            continue
        filtered_headers[header_name] = header_value

    return filtered_headers or None


class Model(BaseModel):
    """Model definition for use without persisting to the Models API."""

    model_config = ConfigDict(extra="forbid", json_schema_extra=_strip_internal_fields)

    url: str = Field(description="URL of the model.")
    name: str = Field(description="Name of the model.")
    default_headers: dict[str, str] | None = Field(
        default=None,
        exclude=True,
        description="Runtime-only non-auth headers automatically applied to requests made with this model. "
        "Authentication must be configured via model.api_key_secret; auth headers such as Authorization will be rejected.",
    )
    host_url: str | None = Field(
        default=None,
        description="Direct NIM endpoint URL (http://host:port). Populated when resolved from a ModelRef. "
        "Used by EvalFactory containers that reject path-based URLs (e.g., Haystack NvidiaDocumentEmbedder).",
    )
    api_key_secret: SecretRef | None = Field(
        default=None,
        description="API key secret reference for the model. Format: workspace/secret_name or secret_name within the job workspace.",
    )
    format: Literal[ModelFormat.NVIDIA_NIM, ModelFormat.OPEN_AI, ModelFormat.LLAMA_STACK] = Field(
        default=ModelFormat.NVIDIA_NIM, description="API format of the model."
    )

    @field_validator("default_headers")
    @classmethod
    def validate_default_headers(cls, value: dict[str, str] | None) -> dict[str, str] | None:
        """Reject auth-style headers and direct users to api_key_secret for credentials."""
        if value is None:
            return None

        for header_name in value:
            if is_auth_header_name(header_name):
                raise ValueError(
                    f"Header {header_name} in model.default_headers is rejected because model.default_headers cannot include authentication headers (Authorization, X-API-Key, etc.). "
                    f"Header names are not allowed to contain the following substrings: {', '.join(_AUTH_HEADER_PATTERNS)}. "
                    f"Configure model auth via model.api_key_secret instead."
                )
        return value

    @cached_property
    def api_key_env(self) -> str | None:
        if self.api_key_secret:
            env_name = self.api_key_secret.root
            if env_name[0].isdigit():
                env_name = f"_{env_name}"  # prefix with valid character for environment variable
            return env_name.replace("-", "_").replace("/", "_")

    @cached_property
    def api_key(self) -> str | None:
        if self.api_key_secret:
            # Env var name uses underscores (shell env vars can't have hyphens)
            api_key_env = self.api_key_env
            assert api_key_env is not None
            # service uses lowercase env, local uses uppercase env
            # TODO: relax Model.api_key_env regex to allow uppercase letters
            if os.getenv(api_key_env) is not None:
                return os.getenv(api_key_env)
            return os.getenv(api_key_env.upper())

    def with_default_headers(self, headers: dict[str, str] | None) -> "Model":
        """Return a copy of the model with merged runtime default headers."""
        if not headers:
            return self

        return self.model_copy(
            update={
                "default_headers": {
                    **(self.default_headers or {}),
                    **headers,
                }
            },
            # flat dict, no deep copy is needed
            deep=False,
        )
