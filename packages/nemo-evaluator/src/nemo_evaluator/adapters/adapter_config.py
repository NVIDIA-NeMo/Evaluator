# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Submodule responsible for the configuration related specifically to adapters.

For the visibility reasons, we don't expose adapter configuration via CLI. All the
adaptor config comes from the framework configuration yaml under
```yaml
target:
  api_endpoint:
    adapter_config:
      discovery:
        modules: ["mod.a.b.c", ...]
        dirs: ["/some/path"]
      use_response_logging: true
      use_reasoning: true
      use_nvcf: true
      use_request_caching: true
      caching_dir: /some/dir
```

This module merely takes such a dict and translates it into a typed dataclass.
"""

from typing import Any

from pydantic import BaseModel, Field

from nemo_evaluator.adapters.reports.post_eval_report_hook import ReportType


class DiscoveryConfig(BaseModel):
    """Configuration for discovering 3rd party modules and directories"""

    modules: list[str] = Field(
        description="List of module paths to discover",
        default_factory=list,
    )
    dirs: list[str] = Field(
        description="List of directory paths to discover",
        default_factory=list,
    )


class InterceptorConfig(BaseModel):
    """Configuration for a single interceptor"""

    name: str = Field(description="Name of the interceptor to use")
    enabled: bool = Field(
        description="Whether this interceptor is enabled", default=True
    )
    config: dict[str, Any] = Field(
        description="Configuration for the interceptor", default_factory=dict
    )


class PostEvalHookConfig(BaseModel):
    """Configuration for a single post-evaluation hook"""

    name: str = Field(description="Name of the post-evaluation hook to use")
    enabled: bool = Field(
        description="Whether this post-evaluation hook is enabled", default=True
    )
    config: dict[str, Any] = Field(
        description="Configuration for the post-evaluation hook", default_factory=dict
    )


class AdapterConfig(BaseModel):
    """Adapter configuration with registry-based interceptor support"""

    discovery: DiscoveryConfig = Field(
        description="Configuration for discovering 3rd party modules and directories",
        default_factory=DiscoveryConfig,
    )
    interceptors: list[InterceptorConfig] = Field(
        description="List of interceptors to use with their configurations",
        default_factory=list,
    )
    post_eval_hooks: list[PostEvalHookConfig] = Field(
        description="List of post-evaluation hooks to use with their configurations",
        default_factory=list,
    )
    endpoint_type: str = Field(
        description="Type of the endpoint to run the adapter for",
        default="chat",
    )
    caching_dir: str | None = Field(
        description="Directory for caching responses (legacy field)",
        default=None,
    )
    generate_html_report: bool = Field(
        description="Whether to generate HTML report (legacy field)",
        default=False,
    )
    log_failed_requests: bool = Field(
        description="Whether to log failed requests (legacy field)",
        default=False,
    )

    @classmethod
    def get_validated_config(cls, run_config: dict[str, Any]) -> "AdapterConfig | None":
        """Extract and validate adapter configuration from run_config.

        Args:
            run_config: The run configuration dictionary

        Returns:
            AdapterConfig instance if adapter_config is present in run_config,
            None otherwise

        Raises:
            ValueError: If adapter_config is present but invalid
        """

        def merge_discovery(
            global_discovery: dict[str, Any], local_discovery: dict[str, Any]
        ) -> dict[str, Any]:
            """Merge global and local discovery configs."""
            return {
                "modules": global_discovery.get("modules", [])
                + local_discovery.get("modules", []),
                "dirs": global_discovery.get("dirs", [])
                + local_discovery.get("dirs", []),
            }

        global_cfg = run_config.get("global_adapter_config", {})
        local_cfg = (
            run_config.get("target", {}).get("api_endpoint", {}).get("adapter_config")
        )

        if not global_cfg and not local_cfg:
            return None

        merged = dict(global_cfg) if global_cfg else {}
        if local_cfg:
            local_discovery = local_cfg.get("discovery")
            global_discovery = merged.get("discovery")
            if local_discovery and global_discovery:
                merged["discovery"] = merge_discovery(global_discovery, local_discovery)
                # Add/override other local fields
                for k, v in local_cfg.items():
                    if k != "discovery":
                        merged[k] = v
            else:
                merged.update(local_cfg)

        # Syntactic sugar, we allow `interceptors` list in non-typed (pre-validation)
        # `adapter_config` to contain also plain strings, which will be treated
        # as `name: <this string>`
        if isinstance(merged.get("interceptors"), list):
            merged["interceptors"] = [
                {"name": s} if isinstance(s, str) else s for s in merged["interceptors"]
            ]

        # Syntactic sugar for post_eval_hooks as well
        if isinstance(merged.get("post_eval_hooks"), list):
            merged["post_eval_hooks"] = [
                {"name": s} if isinstance(s, str) else s
                for s in merged["post_eval_hooks"]
            ]

        try:
            config = cls(**merged)

            # If no interceptors are configured, try to convert from legacy format
            if not config.interceptors:
                config = cls.from_legacy_config(merged)

            return config
        except Exception as e:
            raise ValueError(f"Invalid adapter configuration: {e}") from e

    @classmethod
    def from_legacy_config(cls, legacy_config: dict[str, Any]) -> "AdapterConfig":
        """Convert legacy configuration to new interceptor-based format.

        Args:
            legacy_config: Legacy configuration dictionary

        Returns:
            AdapterConfig instance with interceptors based on legacy config
        """
        interceptors = []
        post_eval_hooks = []

        # Add system message interceptor if custom system prompt is specified (Request)
        if legacy_config.get("use_system_prompt") and legacy_config.get(
            "custom_system_prompt"
        ):
            interceptors.append(
                InterceptorConfig(
                    name="system_message",
                    enabled=True,
                    config={
                        "system_message": legacy_config.get("custom_system_prompt"),
                    },
                )
            )

        # Add payload modifier interceptor if any payload modification parameters are specified (RequestToResponse)
        params_to_add = legacy_config.get("params_to_add")
        params_to_remove = legacy_config.get("params_to_remove")
        params_to_rename = legacy_config.get("params_to_rename")

        if params_to_add or params_to_remove or params_to_rename:
            config = {}
            if params_to_add:
                config["params_to_add"] = params_to_add
            if params_to_remove:
                config["params_to_remove"] = params_to_remove
            if params_to_rename:
                config["params_to_rename"] = params_to_rename

            interceptors.append(
                InterceptorConfig(
                    name="payload_modifier",
                    enabled=True,
                    config=config,
                )
            )

        # Add omni info interceptor if specified (Request)
        if legacy_config.get("use_omni_info"):
            interceptors.append(
                InterceptorConfig(
                    name="omni_info",
                    enabled=True,
                    config={
                        "output_dir": legacy_config.get("output_dir"),
                    },
                )
            )

        # Convert legacy fields to interceptors (Request)
        if legacy_config.get("use_request_logging"):
            config = {"output_dir": legacy_config.get("output_dir")}
            if legacy_config.get("max_logged_requests"):
                config["max_requests"] = legacy_config.get("max_logged_requests")
            interceptors.append(
                InterceptorConfig(
                    name="request_logging",
                    config=config,
                )
            )

        # Add caching interceptor (RequestToResponse)
        if legacy_config.get("use_caching"):
            cache_dir = legacy_config.get("caching_dir", "cache")
            config = {
                "cache_dir": cache_dir,
                "reuse_cached_responses": legacy_config.get(
                    "reuse_cached_responses", True
                ),
                "save_responses": legacy_config.get("save_responses", True),
            }
            if legacy_config.get("save_requests"):
                config["save_requests"] = legacy_config.get("save_requests")
            if legacy_config.get("max_saved_requests"):
                config["max_saved_requests"] = legacy_config.get("max_saved_requests")
            if legacy_config.get("max_saved_responses"):
                config["max_saved_responses"] = legacy_config.get("max_saved_responses")
            interceptors.append(
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config=config,
                )
            )

        # Add the final request interceptor - either nvcf or endpoint
        if legacy_config.get("use_nvcf"):
            interceptors.append(
                InterceptorConfig(
                    name="nvcf",
                    enabled=True,
                    config={},
                )
            )
        else:
            # Only add endpoint if nvcf is not used
            interceptors.append(InterceptorConfig(name="endpoint"))

        if legacy_config.get("use_response_logging"):
            config = {"output_dir": legacy_config.get("output_dir")}
            if legacy_config.get("max_logged_responses"):
                config["max_responses"] = legacy_config.get("max_logged_responses")
            interceptors.append(
                InterceptorConfig(
                    name="response_logging",
                    config=config,
                )
            )

        if legacy_config.get("use_reasoning"):
            config = {
                "end_reasoning_token": legacy_config.get(
                    "end_reasoning_token", "</think>"
                ),
            }
            if legacy_config.get("start_reasoning_token") is not None:
                config["start_reasoning_token"] = legacy_config.get(
                    "start_reasoning_token"
                )
            if legacy_config.get("include_if_reasoning_not_finished") is not None:
                config["include_if_not_finished"] = legacy_config.get(
                    "include_if_reasoning_not_finished"
                )
            if legacy_config.get("track_reasoning") is not None:
                config["enable_reasoning_tracking"] = legacy_config.get(
                    "track_reasoning"
                )
            interceptors.append(
                InterceptorConfig(
                    name="reasoning",
                    config=config,
                )
            )
        if legacy_config.get("use_progress_tracking") or legacy_config.get(
            "output_dir"
        ):
            interceptors.append(
                InterceptorConfig(
                    name="progress_tracking",
                    config={
                        "progress_tracking_url": legacy_config.get(
                            "progress_tracking_url"
                        ),
                        "progress_tracking_interval": legacy_config.get(
                            "progress_tracking_interval", 1
                        ),
                        "output_dir": legacy_config.get("output_dir"),
                    },
                )
            )
            post_eval_hooks.append(
                PostEvalHookConfig(
                    name="progress_tracking",
                    config={
                        "progress_tracking_url": legacy_config.get(
                            "progress_tracking_url"
                        ),
                        "progress_tracking_interval": legacy_config.get(
                            "progress_tracking_interval", 1
                        ),
                        "output_dir": legacy_config.get("output_dir"),
                    },
                )
            )

        # Convert legacy HTML report generation to post-eval hook
        if legacy_config.get("generate_html_report"):
            report_types = [ReportType.HTML]
            if legacy_config.get("include_json", True):
                report_types.append(ReportType.JSON)

            post_eval_hooks.append(
                PostEvalHookConfig(
                    name="post_eval_report",
                    enabled=True,
                    config={
                        "report_types": report_types,
                    },
                )
            )

        return cls(
            interceptors=interceptors,
            post_eval_hooks=post_eval_hooks,
            endpoint_type=legacy_config.get("endpoint_type", "chat"),
            caching_dir=legacy_config.get("caching_dir"),
            generate_html_report=legacy_config.get("generate_html_report", False),
            log_failed_requests=legacy_config.get("log_failed_requests", False),
        )

    def get_interceptor_configs(self) -> dict[str, dict[str, Any]]:
        """Get interceptor configurations as a dictionary"""
        return {ic.name: ic.config for ic in self.interceptors if ic.enabled}

    def get_post_eval_hook_configs(self) -> dict[str, dict[str, Any]]:
        """Get post-evaluation hook configurations as a dictionary"""
        return {hook.name: hook.config for hook in self.post_eval_hooks if hook.enabled}
