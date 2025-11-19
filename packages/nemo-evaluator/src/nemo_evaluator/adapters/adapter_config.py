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

from typing import Any

from pydantic import BaseModel, Field


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

    class Config:
        use_enum_values = True


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
    log_failed_requests: bool = Field(
        description="Whether to log failed requests",
        default=False,
    )

    @classmethod
    def get_legacy_defaults(cls) -> dict[str, Any]:
        """Get default values for legacy configuration parameters."""
        return {
            "generate_html_report": True,
            "html_report_size": 5,
            "tracking_requests_stats": True,
            "caching_dir": None,
            "log_failed_requests": cls.model_fields["log_failed_requests"].default,
            "endpoint_type": cls.model_fields["endpoint_type"].default,
            # Boolean defaults for optional features
            "use_caching": True,
            "save_responses": False,
            "save_requests": False,
            "use_system_prompt": False,
            "use_omni_info": False,
            "use_request_logging": False,
            "use_nvcf": False,
            "use_response_logging": False,
            "use_reasoning": False,
            "process_reasoning_traces": False,
            "use_progress_tracking": False,
            "use_raise_client_errors": False,
            "include_json": True,
            "custom_system_prompt": None,
            "output_dir": None,
            "params_to_add": None,
            "params_to_remove": None,
            "params_to_rename": None,
            "max_logged_requests": None,
            "max_logged_responses": None,
            "max_saved_requests": None,
            "max_saved_responses": None,
            "start_reasoning_token": None,
            "include_if_reasoning_not_finished": None,
            "track_reasoning": None,
            "end_reasoning_token": "</think>",
            "progress_tracking_url": None,
            "progress_tracking_interval": 1,
            "logging_aggregated_stats_interval": 100,
        }

    @classmethod
    def get_validated_config(cls, run_config: dict[str, Any]) -> "AdapterConfig":
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

        # Validate that legacy parameters are not mixed with interceptors
        legacy_defaults = cls.get_legacy_defaults()
        model_fields = set(cls.model_fields.keys())
        legacy_only_params = set(legacy_defaults.keys()) - model_fields

        for config_name, config in [
            ("global_adapter_config", global_cfg),
            ("target.api_endpoint.adapter_config", local_cfg),
        ]:
            if config and config.get("interceptors"):
                found_legacy = [p for p in legacy_only_params if p in config]
                if found_legacy:
                    raise ValueError(
                        f"Cannot use legacy configuration parameters when interceptors are explicitly defined in {config_name}. "
                        f"Found: {', '.join(sorted(found_legacy))}. "
                        f"Please remove these and configure using interceptors instead."
                    )

        if not global_cfg and not local_cfg:
            # Create default adapter config with caching enabled by default
            return cls.from_legacy_config({}, run_config)
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
                config = cls.from_legacy_config(merged, run_config)

            return config
        except Exception as e:
            raise ValueError(f"Invalid adapter configuration: {e}") from e

    @staticmethod
    def _get_default_output_dir(
        legacy_config: dict[str, Any], run_config: dict[str, Any] | None = None
    ) -> str | None:
        """Get default output directory based on configuration priority.

        Args:
            legacy_config: Legacy configuration dictionary
            run_config: Full run configuration dictionary (optional)

        Returns:
            output directory path based on priority: legacy_config.output_dir > run_config.config.output_dir > None
        """
        # First try legacy_config, but handle KeyError if not present
        output_dir = legacy_config.get("output_dir")
        if output_dir is None and run_config:
            output_dir = run_config.get("config", {}).get("output_dir")
        return output_dir

    @staticmethod
    def _get_default_cache_dir(
        legacy_config: dict[str, Any],
        run_config: dict[str, Any] | None = None,
        subdir: str = "cache",
    ) -> str:
        """Get default cache directory based on configuration priority.

        Args:
            legacy_config: Legacy configuration dictionary
            run_config: Full run configuration dictionary (optional)
            subdir: Subdirectory name to append to output_dir (default: "cache")

        Returns:
            cache directory path based on priority: caching_dir > output_dir/{subdir} > /tmp/{subdir}
        """
        # First try caching_dir from legacy config
        cache_dir = legacy_config["caching_dir"]
        if cache_dir is not None:
            return f"{cache_dir}/{subdir}"

        # Fallback to output_dir/{subdir}
        output_dir = AdapterConfig._get_default_output_dir(legacy_config, run_config)
        if output_dir:
            return f"{output_dir}/{subdir}"

        # Final fallback to /tmp/{subdir}
        return f"/tmp/{subdir}"

    @classmethod
    def from_legacy_config(
        cls, legacy_config: dict[str, Any], run_config: dict[str, Any] | None = None
    ) -> "AdapterConfig":
        """Convert legacy configuration to new interceptor-based format.

        Args:
            legacy_config: Legacy configuration dictionary
            run_config: Full run configuration dictionary (optional, used to extract output_dir)

        Returns:
            AdapterConfig instance with interceptors based on legacy config
        """
        # Merge legacy config with defaults to avoid repeated .get() calls
        defaults = cls.get_legacy_defaults()
        legacy_config = {**defaults, **legacy_config}

        interceptors = []
        post_eval_hooks = []

        # Add system message interceptor if custom system prompt is specified (Request)
        if (
            legacy_config["use_system_prompt"]
            and legacy_config["custom_system_prompt"] is not None
        ):
            interceptors.append(
                InterceptorConfig(
                    name="system_message",
                    enabled=True,
                    config={
                        "system_message": legacy_config["custom_system_prompt"],
                    },
                )
            )

        # Add payload modifier interceptor if any payload modification parameters are specified (RequestToResponse)
        params_to_add = legacy_config["params_to_add"]
        params_to_remove = legacy_config["params_to_remove"]
        params_to_rename = legacy_config["params_to_rename"]

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
        if legacy_config["use_omni_info"]:
            interceptors.append(
                InterceptorConfig(
                    name="omni_info",
                    enabled=True,
                    config={
                        "output_dir": cls._get_default_output_dir(
                            legacy_config, run_config
                        ),
                    },
                )
            )

        # Convert legacy fields to interceptors (Request)
        if legacy_config["use_request_logging"]:
            config = {
                "output_dir": cls._get_default_output_dir(legacy_config, run_config)
            }
            if legacy_config["max_logged_requests"] is not None:
                config["max_requests"] = legacy_config["max_logged_requests"]
            interceptors.append(
                InterceptorConfig(
                    name="request_logging",
                    config=config,
                )
            )

        # Add caching interceptor (RequestToResponse)
        # Activate if ANY of these are set: reuse_cached, save_responses, save_requests, generate_html_report
        # For caching interceptor, use caching_dir directly if provided, otherwise use output_dir/cache
        if legacy_config["caching_dir"] is not None:
            cache_dir = legacy_config["caching_dir"]
        else:
            # Use output_dir/cache if output_dir exists, otherwise /tmp/cache
            output_dir = AdapterConfig._get_default_output_dir(
                legacy_config, run_config
            )
            if output_dir:
                cache_dir = f"{output_dir}/cache"
            else:
                cache_dir = "/tmp/cache"

        # Values are now available directly from legacy_config (merged with defaults)
        generate_html_report = legacy_config["generate_html_report"]
        max_html_report_size = legacy_config["html_report_size"]

        # Check if caching should be activated
        should_activate = any(
            [
                legacy_config["use_caching"],
                legacy_config["save_responses"],
                legacy_config["save_requests"],
                generate_html_report,
            ]
        )

        if should_activate:
            # Determine save settings based on generate_html_report
            if generate_html_report:
                save_requests = True
                save_responses = True
                if max_html_report_size is not None:
                    # Handle None values in max() by filtering them out
                    max_saved_requests_values = [max_html_report_size]
                    max_saved_responses_values = [max_html_report_size]

                    if legacy_config["max_saved_requests"] is not None:
                        max_saved_requests_values.append(
                            legacy_config["max_saved_requests"]
                        )
                    if legacy_config["max_saved_responses"] is not None:
                        max_saved_responses_values.append(
                            legacy_config["max_saved_responses"]
                        )

                    max_saved_requests = max(max_saved_requests_values)
                    max_saved_responses = max(max_saved_responses_values)
                else:
                    max_saved_requests = legacy_config["max_saved_requests"]
                    max_saved_responses = legacy_config["max_saved_responses"]
            else:
                save_requests = legacy_config["save_requests"]
                save_responses = legacy_config["save_responses"]
                max_saved_requests = legacy_config["max_saved_requests"]
                max_saved_responses = legacy_config["max_saved_responses"]

            config = {
                "cache_dir": cache_dir,
                "reuse_cached_responses": legacy_config["use_caching"],
                "save_requests": save_requests,
                "save_responses": save_responses,
            }

            if max_saved_requests is not None:
                config["max_saved_requests"] = max_saved_requests
            if max_saved_responses is not None:
                config["max_saved_responses"] = max_saved_responses

            interceptors.append(
                InterceptorConfig(
                    name="caching",
                    enabled=True,
                    config=config,
                )
            )

        # Add the final request interceptor - either nvcf or endpoint
        if legacy_config["use_nvcf"]:
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

        # Add response stats interceptor right after endpoint if tracking is enabled
        # Default to True if not explicitly set in legacy config
        if legacy_config["tracking_requests_stats"]:
            # Use caching_dir if provided, otherwise use output_dir/response_stats_cache
            cache_dir = cls._get_default_cache_dir(
                legacy_config, run_config, "response_stats_cache"
            )
            config = {
                "cache_dir": cache_dir,
                "logging_aggregated_stats_interval": legacy_config[
                    "logging_aggregated_stats_interval"
                ],
            }
            interceptors.append(
                InterceptorConfig(
                    name="response_stats",
                    enabled=True,
                    config=config,
                )
            )

        if legacy_config["use_response_logging"]:
            config = {
                "output_dir": cls._get_default_output_dir(legacy_config, run_config)
            }
            if legacy_config["max_logged_responses"] is not None:
                config["max_responses"] = legacy_config["max_logged_responses"]
            interceptors.append(
                InterceptorConfig(
                    name="response_logging",
                    config=config,
                )
            )

        if legacy_config["use_reasoning"]:
            from nemo_evaluator.logging import get_logger

            logger = get_logger(__name__)
            logger.warning(
                '"use_reasoning" is deprecated as it might suggest it touches on switching on/off reasoning for mode when it does not. Use "process_reasoning_traces" instead.'
            )
            # since we aim at parity between process_reasoning_traces and use_reasoning during deprecation period:
            legacy_config["process_reasoning_traces"] = legacy_config["use_reasoning"]

        if legacy_config["process_reasoning_traces"]:
            # give parity back
            legacy_config["use_reasoning"] = legacy_config["process_reasoning_traces"]
            config = {
                "end_reasoning_token": legacy_config["end_reasoning_token"],
            }
            if legacy_config["start_reasoning_token"] is not None:
                config["start_reasoning_token"] = legacy_config["start_reasoning_token"]
            if legacy_config["include_if_reasoning_not_finished"] is not None:
                config["include_if_not_finished"] = legacy_config[
                    "include_if_reasoning_not_finished"
                ]
            if legacy_config["track_reasoning"] is not None:
                config["enable_reasoning_tracking"] = legacy_config["track_reasoning"]

            # Enable caching for reasoning interceptor when tracking requests stats
            # Default to True if not explicitly set in legacy config
            if legacy_config["tracking_requests_stats"]:
                config["save_individuals"] = True
                # Use caching_dir if provided, otherwise use output_dir/reasoning_stats_cache
                cache_dir = cls._get_default_cache_dir(
                    legacy_config, run_config, "reasoning_stats_cache"
                )
                config["cache_dir"] = cache_dir

            # Add logging interval for aggregated stats
            config["logging_aggregated_stats_interval"] = legacy_config[
                "logging_aggregated_stats_interval"
            ]

            interceptors.append(
                InterceptorConfig(
                    name="reasoning",
                    config=config,
                )
            )
        if legacy_config["use_progress_tracking"]:
            config = {
                "progress_tracking_interval": legacy_config[
                    "progress_tracking_interval"
                ],
                "request_method": "POST",  # Legacy method uses POST
                "output_dir": cls._get_default_output_dir(legacy_config, run_config),
            }
            if legacy_config["progress_tracking_url"] is not None:
                config["progress_tracking_url"] = legacy_config["progress_tracking_url"]

            interceptors.append(
                InterceptorConfig(
                    name="progress_tracking",
                    config=config,
                )
            )

        # Add raise client errors interceptor if specified (Response)
        if legacy_config["use_raise_client_errors"]:
            # Get default values from the interceptor's Params class
            from nemo_evaluator.adapters.interceptors.raise_client_error_interceptor import (
                RaiseClientErrorInterceptor,
            )
            from nemo_evaluator.logging import get_logger

            logger = get_logger(__name__)

            default_params = RaiseClientErrorInterceptor.Params()

            interceptors.append(
                InterceptorConfig(
                    name="raise_client_errors",
                    enabled=True,
                    config={},
                )
            )
            logger.warning(
                "RaiseClientErrorInterceptor configured with default values. "
                f"This will raise exceptions for 4xx status codes ({default_params.status_code_range_start}-{default_params.status_code_range_end}) "
                f"excluding {default_params.exclude_status_codes}. "
                "Consider explicitly configuring the interceptor parameters for your specific use case."
            )

        # Convert legacy HTML report generation to post-eval hook
        # Value is now available directly from legacy_config (merged with defaults)
        generate_html_report = legacy_config["generate_html_report"]
        if generate_html_report:
            report_types = ["html"]
            if legacy_config["include_json"]:
                report_types.append("json")

            post_eval_hooks.append(
                PostEvalHookConfig(
                    name="post_eval_report",
                    enabled=True,
                    config={
                        "report_types": report_types,
                        "html_report_size": legacy_config["html_report_size"],
                    },
                )
            )

        return cls(
            interceptors=interceptors,
            post_eval_hooks=post_eval_hooks,
            endpoint_type=legacy_config["endpoint_type"],
            log_failed_requests=legacy_config["log_failed_requests"],
        )

    def get_interceptor_configs(self) -> dict[str, dict[str, Any]]:
        """Get interceptor configurations as a dictionary"""
        return {ic.name: ic.config for ic in self.interceptors if ic.enabled}

    def get_post_eval_hook_configs(self) -> dict[str, dict[str, Any]]:
        """Get post-evaluation hook configurations as a dictionary"""
        return {hook.name: hook.config for hook in self.post_eval_hooks if hook.enabled}
