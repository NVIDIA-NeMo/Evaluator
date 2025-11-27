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

"""Reasoning interceptor that strips reasoning from responses and tracks reasoning information."""

import json
import re
import threading
from pathlib import Path
from typing import Optional, final

from pydantic import Field

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterResponse,
    PostEvalHook,
    ResponseInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@final
@register_for_adapter(
    name="reasoning",
    description="Processes reasoning content in API responses",
)
class ResponseReasoningInterceptor(ResponseInterceptor, PostEvalHook):
    """Processes reasoning tokens from response. Collects statistics. Strips and/or moves reasoning tokens."""

    class Params(BaseLoggingParams):
        """Configuration parameters for reasoning interceptor."""

        end_reasoning_token: str = Field(
            default="</think>",
            description="Token that marks the end of reasoning section, not used if reasoning_content is provided",
        )
        start_reasoning_token: str | None = Field(
            default="<think>",
            description="Token that marks the start of reasoning section, used for tracking if reasoning has started",
        )
        add_reasoning: bool = Field(
            default=True, description="Whether to add reasoning information"
        )
        migrate_reasoning_content: bool = Field(
            default=False,
            description="If reasoning traces are found in `reasoning_content`, they will be moved to `content` field end surrounded by start_reasoning_token and end_reasoning_token",
        )
        enable_reasoning_tracking: bool = Field(
            default=True, description="Enable reasoning tracking and logging"
        )
        include_if_not_finished: bool = Field(
            default=True,
            description="Include reasoning content if reasoning is not finished (end token not found)",
        )
        stats_file_saving_interval: Optional[int] = Field(
            default=None,
            description="How often (every how many responses) to save stats to a file. If None, stats are only saved via post_eval_hook.",
        )
        enable_caching: bool = Field(
            default=True,
            description="Whether to enable caching of individual request reasoning statistics and aggregated reasoning stats. Useful for resuming interrupted runs.",
        )
        cache_dir: str = Field(
            default="/tmp/reasoning_interceptor",
            description="Custom cache directory for reasoning stats interceptor.",
        )
        logging_aggregated_stats_interval: int = Field(
            default=100,
            description="How often (every how many responses) to log aggregated reasoning statistics. Default is 100.",
        )

    end_reasoning_token: str
    start_reasoning_token: str | None
    add_reasoning: bool
    migrate_reasoning_content: bool
    enable_reasoning_tracking: bool
    include_if_not_finished: bool
    enable_caching: bool
    cache_dir: Optional[str]
    _request_stats_cache: Optional[Cache]
    stats_file_saving_interval: Optional[int]
    logging_aggregated_stats_interval: int

    def __init__(self, params: Params):
        """
        Initialize the reasoning interceptor.

        Args:
            params: Configuration parameters
        """
        self.end_reasoning_token = params.end_reasoning_token
        self.start_reasoning_token = params.start_reasoning_token
        self.add_reasoning = params.add_reasoning
        self.migrate_reasoning_content = params.migrate_reasoning_content
        self.enable_reasoning_tracking = params.enable_reasoning_tracking
        self.include_if_not_finished = params.include_if_not_finished
        self.stats_file_saving_interval = params.stats_file_saving_interval
        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.enable_caching = params.enable_caching
        self.cache_dir = params.cache_dir
        self.logging_aggregated_stats_interval = (
            params.logging_aggregated_stats_interval
        )

        self._lock = threading.Lock()

        # Initialize reasoning stats first, before cache loading
        self._reasoning_stats = {
            "total_responses": 0,
            "responses_with_reasoning": 0,
            "reasoning_finished_count": 0,
            "reasoning_started_count": 0,
            "avg_reasoning_words": None,
            "avg_original_content_words": None,
            "avg_updated_content_words": None,
            "max_reasoning_words": None,
            "max_original_content_words": None,
            "max_updated_content_words": None,
            "max_reasoning_tokens": None,
            "avg_reasoning_tokens": None,
            "max_updated_content_tokens": None,
            "avg_updated_content_tokens": None,
            "total_reasoning_words": 0,
            "total_original_content_words": 0,
            "total_updated_content_words": 0,
            "total_reasoning_tokens": 0,
            "total_updated_content_tokens": 0,
        }

        # Initialize cache if enabled
        if self.enable_caching:
            cache_path = Path(self.cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            self._request_stats_cache = Cache(cache_path)
            self._load_aggregated_cached_stats()
        else:
            self._request_stats_cache = None

        self.logger.info(
            "Reasoning interceptor initialized",
            end_reasoning_token=self.end_reasoning_token,
            start_reasoning_token=self.start_reasoning_token,
            add_reasoning=self.add_reasoning,
            migrate_reasoning_content=self.migrate_reasoning_content,
            enable_reasoning_tracking=self.enable_reasoning_tracking,
            include_if_not_finished=self.include_if_not_finished,
            stats_file_saving_interval=self.stats_file_saving_interval,
            enable_caching=self.enable_caching,
            cache_dir=self.cache_dir,
            logging_aggregated_stats_interval=self.logging_aggregated_stats_interval,
        )

    def _load_aggregated_cached_stats(self) -> None:
        """Load aggregated reasoning stats from cache instead of processing individual requests."""
        if not self._request_stats_cache:
            return

        # Try to load aggregated stats
        try:
            aggregated_data = self._request_stats_cache.get(
                "_aggregated_reasoning_stats"
            )
            if aggregated_data:
                if isinstance(aggregated_data, str):
                    aggregated_data = json.loads(aggregated_data)

                # Restore aggregated stats
                self._reasoning_stats.update(aggregated_data)
                self.logger.info(
                    "Loaded aggregated reasoning stats from cache",
                    total_responses=self._reasoning_stats["total_responses"],
                    responses_with_reasoning=self._reasoning_stats[
                        "responses_with_reasoning"
                    ],
                    max_reasoning_words=self._reasoning_stats["max_reasoning_words"],
                )
                return
        except Exception as e:
            self.logger.warning(f"Failed to load aggregated reasoning stats: {e}")

        # if no aggregated stats, process individual cached requests
        cached_keys = [
            k for k in self._request_stats_cache.iterkeys() if not k.startswith("_")
        ]
        if not cached_keys:
            self.logger.info("No cached reasoning stats found to aggregate")
            return

        self.logger.info(
            f"No aggregated stats found, processing {len(cached_keys)} individual cached requests"
        )

        # Aggregate individual cached stats (fallback for older cache format)
        for key in cached_keys:
            try:
                cached_data = self._request_stats_cache.get(key)
                if cached_data:
                    if isinstance(cached_data, str):
                        cached_data = json.loads(cached_data)
                    self._update_reasoning_stats(cached_data)
            except Exception as e:
                self.logger.warning(
                    f"Failed to process cached reasoning stats for key {key}: {e}"
                )

        # Save the aggregated stats for next time
        self._save_aggregated_stats()

        # Log summary of what was loaded
        self.logger.info(
            "Cached reasoning stats loaded and aggregated",
            total_responses=self._reasoning_stats["total_responses"],
            responses_with_reasoning=self._reasoning_stats["responses_with_reasoning"],
            max_reasoning_words=self._reasoning_stats["max_reasoning_words"],
        )

    def _cache_reasoning_stats(
        self, request_id: str, stats: dict[str, any], context: AdapterGlobalContext
    ) -> None:
        """Cache individual reasoning stats by request ID."""
        if not self.enable_caching:
            return
        try:
            stats_json = json.dumps(stats, ensure_ascii=False)
            self._request_stats_cache[request_id] = stats_json

        except Exception as e:
            self.logger.warning(f"Failed to cache reasoning stats: {e}")

    def _save_aggregated_stats(self) -> None:
        """Save aggregated reasoning stats to cache for efficient loading."""
        if not self.enable_caching or self._request_stats_cache is None:
            self.logger.debug(
                "Skipping aggregated stats save: caching disabled or no cache"
            )
            return

        try:
            stats_json = json.dumps(self._reasoning_stats, ensure_ascii=False)
            self._request_stats_cache["_aggregated_reasoning_stats"] = stats_json
            self.logger.debug(
                "Saved aggregated reasoning stats to cache",
                total_responses=self._reasoning_stats["total_responses"],
                responses_with_reasoning=self._reasoning_stats[
                    "responses_with_reasoning"
                ],
                stats_size_bytes=len(stats_json),
            )
        except Exception as e:
            self.logger.warning(f"Failed to save aggregated reasoning stats: {e}")

    def _update_reasoning_stats(self, reasoning_info: dict) -> None:
        """Update reasoning statistics with new data (thread-safe)."""
        with self._lock:
            self._reasoning_stats["total_responses"] += 1

            # Get values
            reasoning_words = reasoning_info.get("reasoning_words", 0)
            original_words = reasoning_info.get("original_content_words", 0)
            updated_words = reasoning_info.get("updated_content_words", 0)
            reasoning_tokens = reasoning_info.get("reasoning_tokens", "unknown")
            updated_content_tokens = reasoning_info.get(
                "updated_content_tokens", "unknown"
            )

            # Increment counters
            if reasoning_words > 0:
                self._reasoning_stats["responses_with_reasoning"] += 1
            if reasoning_info.get("reasoning_started"):
                self._reasoning_stats["reasoning_started_count"] += 1
            if reasoning_info.get("reasoning_finished"):
                self._reasoning_stats["reasoning_finished_count"] += 1

            # Update running averages
            for stat_key, value in [
                ("avg_reasoning_words", reasoning_words),
                ("avg_original_content_words", original_words),
                ("avg_updated_content_words", updated_words),
                ("avg_reasoning_tokens", reasoning_tokens),
                ("avg_updated_content_tokens", updated_content_tokens),
            ]:
                if value != "unknown":
                    if self._reasoning_stats[stat_key] is None:
                        self._reasoning_stats[stat_key] = value
                    else:
                        self._reasoning_stats[stat_key] = round(
                            (
                                self._reasoning_stats[stat_key]
                                * (self._reasoning_stats["total_responses"] - 1)
                                + value
                            )
                            / self._reasoning_stats["total_responses"],
                            2,
                        )

            # Update max reasoning words
            for key in [
                "reasoning_words",
                "original_content_words",
                "updated_content_words",
                "reasoning_tokens",
                "updated_content_tokens",
            ]:
                value = reasoning_info.get(key, None)
                if value is not None and value != "unknown":
                    if (
                        self._reasoning_stats[f"max_{key}"] is None
                        or value > self._reasoning_stats[f"max_{key}"]
                    ):
                        self._reasoning_stats[f"max_{key}"] = value

            # Update total statistics
            if reasoning_words != "unknown":
                self._reasoning_stats["total_reasoning_words"] += reasoning_words
            if original_words != "unknown":
                self._reasoning_stats["total_original_content_words"] += original_words
            if updated_words != "unknown":
                self._reasoning_stats["total_updated_content_words"] += updated_words
            if reasoning_tokens != "unknown":
                self._reasoning_stats["total_reasoning_tokens"] += reasoning_tokens
            if updated_content_tokens != "unknown":
                self._reasoning_stats["total_updated_content_tokens"] += (
                    updated_content_tokens
                )

            # Log aggregated stats at specified interval
            if (
                self._reasoning_stats["total_responses"]
                % self.logging_aggregated_stats_interval
                == 0
            ):
                self.logger.info(**self._reasoning_stats)

    def _process_reasoning_message(
        self, msg: dict, usage: dict = None
    ) -> tuple[dict, dict]:
        """
        Process reasoning in the message and return modified message with reasoning info.

        Args:
            msg: The message object containing content and potentially reasoning_content
            usage: Optional usage data from the response for token tracking

        Returns:
            tuple: (modified_message, reasoning_info) where reasoning_info has keys:
                   reasoning_words, original_content_words, updated_content_words, reasoning_finished, reasoning_started
        """
        modified_msg = msg.copy()
        content = msg.get("content", "")
        updated_content_tokens = "unknown"
        reasoning_tokens = "unknown"

        # Check if reasoning_content exists in the message and is not empty
        if "reasoning_content" in msg and msg["reasoning_content"] is not None:
            reasoning_content = msg["reasoning_content"]
            updated_message_content = content
            reasoning_started = (
                True
                if reasoning_content is not None and reasoning_content.strip() != ""
                else False
            )
            if content.strip() == "":
                reasoning_finished = False
            else:
                reasoning_finished = True
            if usage:
                # First try to get reasoning_tokens directly from usage
                reasoning_tokens = usage.get("reasoning_tokens", "unknown")
                updated_content_tokens = usage.get("content_tokens", "unknown")

                # If not found, check in completion_tokens_details and output_tokens_details
                if reasoning_tokens == "unknown":
                    for key in ["completion_tokens_details", "output_tokens_details"]:
                        if key in usage:
                            details = usage[key]
                            if isinstance(details, dict):
                                reasoning_tokens = details.get(
                                    "reasoning_tokens", "unknown"
                                )
                                if reasoning_tokens != "unknown":
                                    self.logger.debug(
                                        f"Found reasoning_tokens in {key}: {reasoning_tokens}"
                                    )
                                    break

                # Log if reasoning tokens were found
                if reasoning_tokens != "unknown":
                    self.logger.debug(f"Reasoning tokens extracted: {reasoning_tokens}")
                else:
                    self.logger.debug("No reasoning tokens found in usage data")

        else:
            reasoning_finished = False
            if self.start_reasoning_token is not None:
                reasoning_started = self.start_reasoning_token in content
            else:
                reasoning_started = "unknown"
            if self.end_reasoning_token in content:
                reasoning_finished = True
                reasoning_started = True

            # Split content using reasoning token
            if reasoning_finished:
                cleaned_content = self._strip_reasoning(content)
                reasoning_content = content[: content.find(self.end_reasoning_token)]
                updated_message_content = cleaned_content
            else:
                if reasoning_started == "unknown":
                    reasoning_content = "unknown"
                elif reasoning_started:
                    reasoning_content = content
                else:
                    reasoning_content = ""
                if not self.include_if_not_finished:
                    updated_message_content = ""
                else:
                    updated_message_content = content

        # Assign the updated message content
        modified_msg["content"] = updated_message_content

        # Calculate lengths and reasoning status
        if reasoning_content and reasoning_content != "unknown":
            reasoning_words = len(reasoning_content.split())
        elif reasoning_started == "unknown":
            reasoning_words = "unknown"
        else:
            reasoning_words = 0

        reasoning_info = {
            "reasoning_words": reasoning_words,
            "original_content_words": (len(content.split()) if content else 0),
            "updated_content_words": (
                len(modified_msg.get("content", "").split())
                if modified_msg.get("content")
                else 0
            ),
            "reasoning_finished": reasoning_finished,
            "reasoning_started": reasoning_started,
            "reasoning_tokens": reasoning_tokens,
            "updated_content_tokens": updated_content_tokens,
        }

        return modified_msg, reasoning_info

    def _strip_reasoning(self, text: str) -> str:
        """Remove everything between start and end reasoning tokens."""
        if not isinstance(text, str):
            return ""

        # Remove everything between start and end reasoning tokens
        # Also handle cases where only end token is present
        cleaned_content = re.sub(
            r".*?" + re.escape(self.end_reasoning_token),
            "",
            text,
            flags=re.DOTALL,
        ).strip("\n")
        return cleaned_content

    def _migrate_reasoning_content(self, msg: dict):
        """Migrate reasoning content to the content field with reasoning tokens."""
        modified_msg = msg.copy()
        if (
            "reasoning_content" in msg
            and msg["reasoning_content"]
            and msg["reasoning_content"].strip()
        ):
            reasoning_content = msg["reasoning_content"]
            content = msg.get("content", "")
            updated_message_content = (
                self.start_reasoning_token
                + reasoning_content
                + self.end_reasoning_token
                + content
            )
            modified_msg["content"] = updated_message_content
            return modified_msg
        else:
            return msg

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Remove reasoning tokens from assistant message content in the response and track reasoning info."""
        if not self.add_reasoning:
            if self.migrate_reasoning_content:
                self.logger.debug(
                    "Reasoning processing disabled, but migrating `reasoning_content` back to `content`."
                )
                response_data = resp.r.json()
                for choice in response_data["choices"]:
                    msg = choice.get("message")
                    modified_msg = self._migrate_reasoning_content(msg)
                    msg.update(modified_msg)

                resp.r._content = json.dumps(response_data).encode()
            else:
                self.logger.debug(
                    "Reasoning processing disabled, returning response as-is"
                )
            return resp
        if resp.rctx.cache_hit:
            self.logger.debug("Response was from cache, skipping reasoning processing")
            return resp

        try:
            response_data = resp.r.json()

            if isinstance(response_data, dict) and "choices" in response_data:
                self.logger.debug(
                    "Processing response with choices",
                    choices_count=len(response_data["choices"]),
                )

                # Extract usage data from response
                usage_data = response_data.get("usage", {})

                for choice in response_data["choices"]:
                    msg = choice.get("message")
                    if (
                        msg
                        and msg.get("role") == "assistant"
                        and isinstance(msg.get("content"), str)
                    ):
                        # Get modified message and reasoning information
                        modified_msg, reasoning_info = self._process_reasoning_message(
                            msg, usage_data
                        )

                        # Collect reasoning statistics
                        self._update_reasoning_stats(reasoning_info)

                        # Log reasoning information if tracking is enabled
                        if self.enable_reasoning_tracking:
                            self.logger.info(
                                "Reasoning tracking information", **reasoning_info
                            )

                        # Update the message with the modified content
                        msg.update(modified_msg)

                        # Save stats to file if interval reached
                        if (
                            self.stats_file_saving_interval is not None
                            and self._reasoning_stats["total_responses"]
                            % self.stats_file_saving_interval
                            == 0
                        ):
                            self._save_stats_to_file(context)

                        self.logger.debug(
                            "Message processed",
                            role=msg.get("role"),
                            original_content_length=reasoning_info[
                                "original_content_words"
                            ],
                            updated_content_length=reasoning_info[
                                "updated_content_words"
                            ],
                            reasoning_words=reasoning_info["reasoning_words"],
                        )

                # Cache individual reasoning stats if enabled
                if self.enable_caching:
                    request_id = resp.rctx.request_id
                    if request_id:
                        # Cache the individual reasoning stats
                        self._cache_reasoning_stats(request_id, reasoning_info, context)
                        self.logger.debug(
                            "Cached individual reasoning stats",
                            request_id=request_id,
                            individual_stats=reasoning_info,
                        )
                self._save_aggregated_stats()  # Save aggregated stats periodically

            resp.r._content = json.dumps(response_data).encode()

            self.logger.info(
                "Response reasoning processing completed",
            )

        except Exception as e:
            self.logger.error("Failed to process response reasoning", error=str(e))
            pass

        return resp

    def _save_stats_to_file(self, context: AdapterGlobalContext) -> None:
        """Save current stats to the same file as post-eval hook."""
        # Get stats in a thread-safe manner
        with self._lock:
            stats = self._reasoning_stats.copy()

        # Don't create file if no stats collected
        if stats["total_responses"] == 0:
            self.logger.debug("No reasoning statistics collected, skipping file write")
            return

        # Prepare metrics data under adapter name
        metrics_data = {
            "reasoning": {
                "description": "Reasoning statistics saved during processing",
                **stats,
            }
        }

        with self._lock:
            context.metrics_path.parent.mkdir(parents=True, exist_ok=True)

            # Read existing metrics if file exists
            existing_metrics = {}
            if context.metrics_path.exists():
                try:
                    with open(context.metrics_path, "r") as f:
                        existing_metrics = json.load(f)
                except (json.JSONDecodeError, IOError):
                    pass  # Start fresh if file is corrupted

            # Merge with existing metrics
            merged_metrics = {**existing_metrics, **metrics_data}

            # Write merged metrics to file
            with open(context.metrics_path, "w") as f:
                json.dump(merged_metrics, f, indent=2, ensure_ascii=False)

        self.logger.debug(
            "Saved reasoning stats to file",
            path=str(context.metrics_path),
            response_count=stats["total_responses"],
        )

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Write collected reasoning statistics to eval_factory_metrics.json."""
        self.logger.info(
            "Writing reasoning statistics to metrics",
            total_responses=self._reasoning_stats["total_responses"],
            output_dir=context.output_dir,
        )
        self._save_stats_to_file(context)
