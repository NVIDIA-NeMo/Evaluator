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
"""Service configuration schemas (vLLM, SGLang, NIM, external API, Gym, NAT, custom)."""

from __future__ import annotations

import warnings
from typing import Annotated, Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Discriminator, Field, Tag, field_validator, model_validator

Protocol = Literal["chat_completions", "completions", "responses"]


class GenerationConfig(BaseModel):
    """Default generation parameters for a service.  Solvers may override
    these per-benchmark via their own `generation` field.

    Merge semantics: when a solver specifies generation, each non-None
    field overrides the corresponding service-level field.  Fields left
    as None inherit from the service's generation config."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=None, gt=0)
    seed: int | None = None
    stop: list[str] | None = None
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)

    def merge_onto(self, base: GenerationConfig) -> GenerationConfig:
        """Return a new config with self's non-None fields overriding base."""
        merged = {}
        for field_name in self.model_fields:
            override = getattr(self, field_name)
            merged[field_name] = override if override is not None else getattr(base, field_name)
        return GenerationConfig(**merged)


class InterceptorConfig(BaseModel):
    """An interceptor attached to a service's adapter proxy."""

    model_config = ConfigDict(extra="forbid")

    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class ProxyConfig(BaseModel):
    """Adapter proxy settings for a service.

    ``extra_body`` is merged into every outgoing request body.
    ``request_timeout`` sets the HTTP timeout for upstream requests.
    ``max_retries`` and ``retry_on_status`` control retry behavior.
    ``max_concurrent_upstream`` limits concurrent requests to the upstream.
    """

    model_config = ConfigDict(extra="forbid")

    interceptors: list[InterceptorConfig] = Field(default_factory=list)
    verbose: bool = False
    extra_body: dict[str, Any] = Field(default_factory=dict)
    request_timeout: float = 120.0
    max_retries: int = 0
    retry_on_status: list[int] = Field(default_factory=lambda: [429, 502, 503, 504])
    max_concurrent_upstream: int = 64

    @property
    def needs_proxy(self) -> bool:
        """Whether any proxy-relevant config is present."""
        return bool(self.interceptors or self.extra_body or self.verbose)


class _ModelServerBase(BaseModel):
    """Shared fields for locally-deployed model servers (vllm, sglang,
    nim, docker_model).  NEL starts these servers and knows their URL."""

    model_config = ConfigDict(extra="forbid")

    model: str
    served_model_name: str | None = None
    port: int = 8000
    protocol: Protocol
    tensor_parallel_size: int | None = None
    pipeline_parallel_size: int | None = None
    data_parallel_size: int | None = None
    num_nodes: int = 1
    gpus: list[int] | int | None = None
    image: str | None = None
    health_path: str = "/health"
    startup_timeout: float = 600.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    extra_args: list[str] = Field(default_factory=list)
    setup_commands: list[str] = Field(default_factory=list)
    container_mounts: list[str] = Field(default_factory=list)
    ray_binary: str = "ray"
    reasoning_pattern: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    proxy: ProxyConfig | None = None
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return True

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}/v1"


class VllmService(_ModelServerBase):
    type: Literal["vllm"] = "vllm"


class SglangService(_ModelServerBase):
    type: Literal["sglang"] = "sglang"


class NimService(_ModelServerBase):
    type: Literal["nim"] = "nim"


class DockerModelService(_ModelServerBase):
    type: Literal["docker_model"] = "docker_model"


class ExternalApiService(BaseModel):
    """Pre-deployed model / judge behind an HTTP endpoint.

    `url` is the FULL URL where requests are sent (always include path).
    `protocol` is the wire format (can decouple for experimental endpoints).
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["api"] = "api"
    url: str
    protocol: Protocol
    model: str | None = None
    api_key: str | None = None
    health_path: str | None = None
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    proxy: ProxyConfig | None = None

    @property
    def is_model_server(self) -> bool:
        return True

    @property
    def is_managed(self) -> bool:
        return False

    @property
    def base_url(self) -> str:
        return self.url

    @field_validator("url")
    @classmethod
    def _validate_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Service url must start with http:// or https://, got: {v!r}")
        return v

    @model_validator(mode="after")
    def _warn_url_protocol_mismatch(self) -> ExternalApiService:
        _SUFFIX_TO_PROTOCOL = [
            ("/chat/completions", "chat_completions"),
            ("/completions", "completions"),
            ("/responses", "responses"),
        ]
        path = urlparse(self.url).path
        for suffix, expected in _SUFFIX_TO_PROTOCOL:
            if path.endswith(suffix):
                if self.protocol != expected:
                    warnings.warn(
                        f"Service url '{self.url}' ends with '{suffix}' but "
                        f"protocol is '{self.protocol}' (expected '{expected}').",
                        UserWarning,
                        stacklevel=2,
                    )
                break
        return self


class GymResourceService(BaseModel):
    """A nemo-gym resource server (exposes /seed_session, /verify, tool
    endpoints).  Not a model — no protocol/generation/interceptors."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["gym"] = "gym"
    url: str | None = None
    port: int = 8000
    image: str | None = None
    benchmark: str | None = None
    server_cmd: str | None = None
    startup_timeout: float = 120.0
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return self.url is None

    @property
    def base_url(self) -> str:
        if self.url:
            return self.url
        return f"http://localhost:{self.port}"


class NatAgentService(BaseModel):
    """A NAT agent server."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["nat"] = "nat"
    port: int = 8000
    image: str | None = None
    nat_config_file: str | None = None
    startup_timeout: float = 120.0
    depends_on: list[str] = Field(default_factory=list)
    node_pool: str | None = None

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"


class CustomService(BaseModel):
    """Plugin service — dynamically imported from class_path."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["custom"] = "custom"
    class_path: str
    config: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_model_server(self) -> bool:
        return False

    @property
    def is_managed(self) -> bool:
        return True

    @property
    def base_url(self) -> str:
        return ""


def _service_discriminator(v: Any) -> str:
    if isinstance(v, dict):
        t = v.get("type")
        if t is None:
            raise ValueError("Service config must include a 'type' field")
        return t
    t = getattr(v, "type", None)
    if t is None:
        raise ValueError(f"Cannot determine service type from {type(v).__name__}. Expected a dict with a 'type' field.")
    return t


ServiceConfig = Annotated[
    Annotated[VllmService, Tag("vllm")]
    | Annotated[SglangService, Tag("sglang")]
    | Annotated[NimService, Tag("nim")]
    | Annotated[DockerModelService, Tag("docker_model")]
    | Annotated[ExternalApiService, Tag("api")]
    | Annotated[GymResourceService, Tag("gym")]
    | Annotated[NatAgentService, Tag("nat")]
    | Annotated[CustomService, Tag("custom")],
    Discriminator(_service_discriminator),
]

_MODEL_SERVICE_TYPES = (
    VllmService,
    SglangService,
    NimService,
    DockerModelService,
    ExternalApiService,
)
