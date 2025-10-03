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

"""Response stats interceptor that collects aggregated statistics from API responses."""

import datetime
import json
import threading
import time
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


@register_for_adapter(
    name="response_stats",
    description="Collects aggregated statistics from API responses for metrics collection",
)
@final
class ResponseStatsInterceptor(ResponseInterceptor, PostEvalHook):
    """Collects aggregated statistics from API responses for metrics collection.

    Tracks the following statistics:
    - Token usage (prompt, completion, total) with averages and maximums
    - Response status codes and counts
    - Finish reasons and stop reasons
    - Tool calls and function calls counts
    - Response latency (average and maximum)
    - Total response count
    - Number of runs, inference times (approximated by processing time from the first to the last response)
    """

    class Params(BaseLoggingParams):
        """Configuration parameters for response stats collection."""

        collect_token_stats: bool = Field(
            default=True, description="Whether to collect token statistics"
        )
        collect_finish_reasons: bool = Field(
            default=True, description="Whether to collect finish reasons"
        )
        collect_tool_calls: bool = Field(
            default=True, description="Whether to collect tool call statistics"
        )
        stats_file_saving_interval: Optional[int] = Field(
            default=None,
            description="How often (every how many responses) to save stats to a file. If None, stats are only saved via post_eval_hook.",
        )
        save_individuals: bool = Field(
            default=True,
            description="Whether to save individual request statistics. If True, saves all individuals; if False, saves only aggregated stats.",
        )
        cache_dir: str = Field(
            default="/tmp/response_stats_interceptor",
            description="Custom cache directory for response stats interceptor.",
        )
        logging_aggregated_stats_interval: int = Field(
            default=100,
            description="How often (every how many responses) to log aggregated response statistics. Default is 100.",
        )

    def __init__(self, params: Params):
        """
        Initialize the response stats interceptor.

        Args:
            params: Configuration parameters
        """
        self.collect_token_stats = params.collect_token_stats
        self.collect_finish_reasons = params.collect_finish_reasons
        self.collect_tool_calls = params.collect_tool_calls
        self.stats_file_saving_interval = params.stats_file_saving_interval
        self.save_individuals = params.save_individuals
        self.cache_dir = params.cache_dir
        self.logging_aggregated_stats_interval = (
            params.logging_aggregated_stats_interval
        )
        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        # Initialize lock and stats first
        self._lock = threading.Lock()
        self._adapter_start_time = time.time()  # Record adapter initialization time
        self._stats = {
            # Average statistics
            "avg_prompt_tokens": None,
            "avg_total_tokens": None,
            "avg_completion_tokens": None,
            "avg_latency_ms": None,
            # Maximum statistics
            "max_prompt_tokens": None,
            "max_total_tokens": None,
            "max_completion_tokens": None,
            "max_latency_ms": None,
            # Counters and totals
            "count": 0,
            "successful_count": 0,
            "tool_calls_count": 0,
            "function_calls_count": 0,
            "finish_reason": {},
            "stop_reason": {},
            "status_codes": {},
            # Time tracking
            "inference_time": 0.0,
            "run_id": 0,
            "last_request_time": None,
            "inference_run_times": {},  # {run_id: {"start": time, "end": time, "inference_time": time}}
        }

        # Always initialize cache database
        cache_path = Path(self.cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self._request_stats_cache = Cache(cache_path)

        # Load existing aggregated stats if available
        try:
            self._load_aggregated_cached_stats()
        except Exception as e:
            self.logger.warning(f"Failed to load cached stats: {e}")

        # Save run info immediately on initialization
        self._save_run_ids_info()

        self.logger.info(
            "Response stats interceptor initialized",
            collect_token_stats=self.collect_token_stats,
            collect_finish_reasons=self.collect_finish_reasons,
            collect_tool_calls=self.collect_tool_calls,
            stats_file_saving_interval=self.stats_file_saving_interval,
            save_individuals=self.save_individuals,
            cache_dir=self.cache_dir,
            logging_aggregated_stats_interval=self.logging_aggregated_stats_interval,
        )

    def _load_aggregated_cached_stats(self) -> None:
        """Load interceptor state from cache."""
        interceptor_state = self._load_interceptor_state()

        if "aggregated_stats" in interceptor_state:
            aggregated_stats = interceptor_state["aggregated_stats"]

            # Convert ISO timestamps back to floats for inference_run_times (if they are strings)
            if "inference_run_times" in aggregated_stats:
                # Convert string keys back to integers for run_ids
                converted_run_times = {}
                for run_id, run_data in aggregated_stats["inference_run_times"].items():
                    # Convert run_id from string to int if needed
                    int_run_id = int(run_id) if isinstance(run_id, str) else run_id

                    if run_data.get("run_start") and isinstance(
                        run_data["run_start"], str
                    ):
                        run_data["run_start"] = datetime.datetime.fromisoformat(
                            run_data["run_start"]
                        ).timestamp()
                    if run_data.get("first_request_time") and isinstance(
                        run_data["first_request_time"], str
                    ):
                        run_data["first_request_time"] = (
                            datetime.datetime.fromisoformat(
                                run_data["first_request_time"]
                            ).timestamp()
                        )
                    if run_data.get("last_request_time") and isinstance(
                        run_data["last_request_time"], str
                    ):
                        run_data["last_request_time"] = datetime.datetime.fromisoformat(
                            run_data["last_request_time"]
                        ).timestamp()

                    converted_run_times[int_run_id] = run_data

                # Replace with converted run_times
                aggregated_stats["inference_run_times"] = converted_run_times

            # Convert string keys back to integers for status codes
            if "status_codes" in aggregated_stats and isinstance(
                aggregated_stats["status_codes"], dict
            ):
                status_codes = {}
                for key, value in aggregated_stats["status_codes"].items():
                    try:
                        int_key = int(key)
                        status_codes[int_key] = value
                    except ValueError:
                        status_codes[key] = value
                aggregated_stats["status_codes"] = status_codes

            # Set current stats to cached data (cached stats already contain accumulated data)
            self._stats = aggregated_stats
            # Note: run_id increment is handled in _save_run_ids_info()

            self.logger.info(
                f"Loaded interceptor state with run_id {aggregated_stats.get('run_id', 0)}, count={aggregated_stats.get('count', 0)}"
            )
        else:
            self.logger.info("No cached interceptor state found")

    def _update_basic_stats(self, resp: AdapterResponse, current_time: float) -> None:
        """Update basic statistics with thread safety."""
        with self._lock:
            # Update last_request_time
            self._stats["last_request_time"] = current_time

            # Update inference_run_times for current run
            run_id = self._stats["run_id"]
            if run_id not in self._stats["inference_run_times"]:
                # First request in this run - estimate when inference actually started using latency
                estimated_first_request_start = current_time
                if hasattr(resp, "latency_ms") and resp.latency_ms is not None:
                    # Estimate when this request was sent (current_time - latency)
                    estimated_first_request_start = current_time - (
                        resp.latency_ms / 1000.0
                    )

                self._stats["inference_run_times"][run_id] = {
                    "run_start": self._adapter_start_time,
                    "first_request_time": estimated_first_request_start,
                    "last_request_time": current_time,
                    "inference_time": 0.0,
                }
            else:
                # Update last_request_time and calculate inference_time
                run_data = self._stats["inference_run_times"][run_id]
                old_inference_time = run_data["inference_time"]
                run_data["last_request_time"] = current_time
                run_data["inference_time"] = (
                    current_time - run_data["first_request_time"]
                )

                # Add delta to global inference_time
                delta = run_data["inference_time"] - old_inference_time
                self._stats["inference_time"] += delta

    def _update_running_stats(self, stat_name: str, value: float) -> None:
        """Update running average and max for a given statistic."""
        # Skip if value is not a valid number
        if not isinstance(value, (int, float)):
            self.logger.warning(
                f"Invalid value for {stat_name}: {value} (expected number)"
            )
            return

        # Calculate running average using current successful count
        avg_key = f"avg_{stat_name}"
        if self._stats[avg_key] is None:
            self._stats[avg_key] = value
        else:
            self._stats[avg_key] = round(
                (self._stats[avg_key] * self._stats["successful_count"] + value)
                / (self._stats["successful_count"] + 1),
                2,
            )

        # Update max valuename
        max_key = f"max_{stat_name}"
        if self._stats[max_key] is None or value > self._stats[max_key]:
            self._stats[max_key] = value

    def _update_time_tracking(self, current_time: float) -> None:
        """Update time tracking statistics (thread-safe)."""
        with self._lock:
            # Update last request time
            self._stats["last_request_time"] = current_time

    def _update_response_stats(self, individual_stats: dict[str, any]) -> None:
        """Update response statistics with new data (thread-safe)."""
        with self._lock:
            # Update token statistics with running means BEFORE incrementing successful_count
            for token_type in ["prompt_tokens", "total_tokens", "completion_tokens"]:
                value = individual_stats.get(token_type, 0)
                self._update_running_stats(token_type, value)

            # Increment successful count after updating running averages
            self._stats["successful_count"] += 1

            # Update finish reasons
            finish_reason = individual_stats.get("finish_reason")
            if isinstance(finish_reason, str):
                self._stats["finish_reason"][finish_reason] = (
                    self._stats["finish_reason"].get(finish_reason, 0) + 1
                )

            # Update tool calls and function calls
            tool_calls_count = individual_stats.get("tool_calls_count", 0)
            if tool_calls_count > 0:
                self._stats["tool_calls_count"] += tool_calls_count

            function_calls_count = individual_stats.get("function_calls_count", 0)
            if function_calls_count > 0:
                self._stats["function_calls_count"] += function_calls_count

            # Log aggregated stats at specified interval
            if (
                self._stats["successful_count"] % self.logging_aggregated_stats_interval
                == 0
            ):
                self.logger.info(**self._stats)

    def _add_basic_response_stats(
        self, adapter_response, context: AdapterGlobalContext
    ) -> None:
        """Add basic statistics for any response (JSON or non-JSON)."""
        with self._lock:
            self._stats["count"] += 1

            # Track the specific status code
            status_code = adapter_response.r.status_code
            self._stats["status_codes"][status_code] = (
                self._stats["status_codes"].get(status_code, 0) + 1
            )

            # Track latency statistics
            if (
                hasattr(adapter_response, "latency_ms")
                and adapter_response.latency_ms is not None
            ):
                self._update_running_stats("latency_ms", adapter_response.latency_ms)

    def _extract_detailed_response_stats(self, response_data: dict) -> dict:
        """Extract detailed response statistics from response data."""
        detailed_stats = {}

        try:
            # Extract usage information
            usage = response_data.get("usage", {})
            if isinstance(usage, dict):
                detailed_stats["prompt_tokens"] = usage.get("prompt_tokens", 0)
                detailed_stats["total_tokens"] = usage.get("total_tokens", 0)
                detailed_stats["completion_tokens"] = usage.get("completion_tokens", 0)

            # Extract choices information
            choices = response_data.get("choices", [])
            if isinstance(choices, list):
                for choice in choices:
                    if isinstance(choice, dict):
                        # Track finish reasons
                        finish_reason = choice.get("finish_reason")
                        if isinstance(finish_reason, str):
                            detailed_stats["finish_reason"] = finish_reason

                        # Track tool calls and function calls
                        message = choice.get("message", {})
                        if isinstance(message, dict):
                            tool_calls = message.get("tool_calls", [])
                            if isinstance(tool_calls, list):
                                detailed_stats["tool_calls_count"] = len(tool_calls)

                            function_call = message.get("function_call")
                            detailed_stats["function_calls_count"] = (
                                1 if function_call else 0
                            )
                        break  # Only process first choice for individual stats
        except Exception as e:
            self.logger.warning(
                "Failed to extract detailed response stats",
                error=str(e),
            )

        return detailed_stats

    def _cache_request_stats(self, request_id: str, stats: dict[str, any]) -> None:
        """Cache individual request stats by request ID."""
        # Only save individual requests if save_individuals is True
        if self.save_individuals and self._request_stats_cache is not None:
            # Add request_id to the stats before caching
            stats_with_id = stats.copy()
            stats_with_id["request_id"] = request_id
            stats_json = json.dumps(stats_with_id, ensure_ascii=False)
            self._request_stats_cache.set(request_id, stats_json)

    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Collect aggregated statistics from the response."""
        # Get status code once and reuse it
        if resp.rctx.cache_hit:
            self.logger.debug(
                "Response was from cache, skipping response stats collection"
            )
            return resp
        status_code = resp.r.status_code

        # Update time tracking with current timestamp
        current_time = time.time()
        self._update_time_tracking(current_time)

        # Update basic stats with thread safety
        self._update_basic_stats(resp, current_time)

        # Always add basic response stats (count, status_code)
        self._add_basic_response_stats(resp, context)

        # Extract detailed stats once and reuse them
        detailed_stats = None
        try:
            # Try to parse response as JSON
            response_data = resp.r.json()

            if status_code == 200:
                detailed_stats = self._extract_detailed_response_stats(response_data)

                # Add detailed stats for aggregation
                self._update_response_stats(detailed_stats)

                self.logger.debug(
                    "Collected detailed response stats",
                    request_id=resp.rctx.request_id,
                    response_count=self._stats["count"],
                    status_code=status_code,
                )

        except (json.JSONDecodeError, Exception) as e:
            # Handle both JSON parsing errors and other exceptions
            # In case of any error, only basic stats are collected
            self.logger.warning(f"Error parsing response body for token counting: {e}")

        # Save stats to file if interval reached
        if (
            self.stats_file_saving_interval is not None
            and self._stats["count"] % self.stats_file_saving_interval == 0
        ):
            self._save_stats_to_file(context)

        # Cache individual request stats if enabled
        if self.save_individuals:
            request_id = resp.rctx.request_id
            # Create individual request stats with basic info
            individual_stats = {
                "timestamp": current_time,
                "status_code": status_code,
                "count": 1,  # This is just one response
                "run_id": self._stats["run_id"],
            }

            # Add detailed stats if available (reuse the extracted stats)
            if detailed_stats:
                individual_stats.update(detailed_stats)

            self._cache_request_stats(request_id, individual_stats)

        # Save aggregated stats to cache
        self._save_aggregated_stats_to_cache()

        return resp

    def _save_stats_to_file(self, context: AdapterGlobalContext) -> None:
        """Save current stats to the same file as post-eval hook."""
        # Get stats in a thread-safe manner
        with self._lock:
            stats = self._stats.copy()

        # Don't create file if no stats collected
        if stats["count"] == 0:
            self.logger.debug("No response statistics collected, skipping file write")
            return

        # Convert timestamps to readable dates in inference_run_times and add time_to_first_request
        if "inference_run_times" in stats:
            for run_id, run_data in stats["inference_run_times"].items():
                if run_data.get("run_start"):
                    run_data["run_start"] = datetime.datetime.fromtimestamp(
                        run_data["run_start"]
                    ).isoformat()
                if run_data.get("first_request_time"):
                    run_data["first_request_time"] = datetime.datetime.fromtimestamp(
                        run_data["first_request_time"]
                    ).isoformat()
                if run_data.get("last_request_time"):
                    run_data["last_request_time"] = datetime.datetime.fromtimestamp(
                        run_data["last_request_time"]
                    ).isoformat()

                # Calculate time_to_first_request for this run
                if run_data.get("first_request_time") and run_data.get("run_start"):
                    # Convert ISO strings back to timestamps for calculation
                    first_request_timestamp = datetime.datetime.fromisoformat(
                        run_data["first_request_time"]
                    ).timestamp()
                    run_start_timestamp = datetime.datetime.fromisoformat(
                        run_data["run_start"]
                    ).timestamp()
                    time_to_first_request = (
                        first_request_timestamp - run_start_timestamp
                    )
                    run_data["time_to_first_request_seconds"] = round(
                        time_to_first_request, 3
                    )

        # Prepare metrics data under adapter name
        metrics_data = {
            "response_stats": {
                "description": "Response statistics saved during processing",
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
            "Saved response stats to file",
            path=str(context.metrics_path),
            response_count=stats["count"],
        )

    def _save_run_ids_info(self) -> None:
        """Save run IDs info during initialization as dictionary in cache."""
        # Load existing interceptor state
        interceptor_state = self._load_interceptor_state()

        # Determine the next run_id based on existing run_ids
        if "run_ids" in interceptor_state and interceptor_state["run_ids"]:
            # Get the highest run_id and increment it
            max_run_id = max(run["run_id"] for run in interceptor_state["run_ids"])
            self._stats["run_id"] = max_run_id + 1
        else:
            # First run, start with 0
            self._stats["run_id"] = 0

        run_info = {
            "run_id": self._stats["run_id"],
            "start": datetime.datetime.fromtimestamp(
                self._adapter_start_time
            ).isoformat(),
        }

        # Add current run info
        if "run_ids" not in interceptor_state:
            interceptor_state["run_ids"] = []
        interceptor_state["run_ids"].append(run_info)

        # Save updated interceptor state
        self._save_interceptor_state(interceptor_state)

        self.logger.debug(
            "Saved run info to interceptor state",
            run_id=self._stats["run_id"],
            start_time=run_info["start"],
        )

    def _save_aggregated_stats_to_cache(self) -> None:
        """Save aggregated stats to interceptor state."""
        # Load existing interceptor state
        interceptor_state = self._load_interceptor_state()

        # Create a copy of stats for caching
        stats_to_cache = self._stats.copy()

        # Keep timestamps as floats for caching (don't convert to ISO)

        # Update aggregated stats in interceptor state
        interceptor_state["aggregated_stats"] = stats_to_cache

        # Save updated interceptor state
        self._save_interceptor_state(interceptor_state)

        self.logger.debug(
            "Saved aggregated stats to interceptor state",
            run_id=self._stats["run_id"],
        )

    def _load_interceptor_state(self) -> dict:
        """Load interceptor state from cache."""
        if self._request_stats_cache is not None:
            state = self._request_stats_cache.get("interceptor_state")
            if state:
                if isinstance(state, str):
                    return json.loads(state)
                return state
        return {}

    def _save_interceptor_state(self, state: dict) -> None:
        """Save interceptor state to cache."""
        if self._request_stats_cache is not None:
            self._request_stats_cache.set("interceptor_state", state)

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Write collected response statistics to eval_factory_metrics.json."""
        # Get aggregated stats
        self.logger.info(
            "Writing response statistics to metrics",
            total_responses=self._stats["count"],
            successful_responses=self._stats["successful_count"],
            output_dir=context.output_dir,
        )
        self._save_stats_to_file(context)
