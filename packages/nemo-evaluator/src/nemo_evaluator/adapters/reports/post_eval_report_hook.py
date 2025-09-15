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

"""Post-evaluation report generation hook."""

import json
from pathlib import Path
from typing import Any, List, Literal

from jinja2 import Environment, StrictUndefined, select_autoescape
from pydantic import BaseModel, Field

from nemo_evaluator.adapters.caching.diskcaching import Cache
from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.reports.templates.simple_template import SIMPLE_TEMPLATE
from nemo_evaluator.adapters.types import AdapterGlobalContext, PostEvalHook
from nemo_evaluator.logging import get_logger


@register_for_adapter(
    name="post_eval_report",
    description="Generates reports of cached requests and responses",
)
class PostEvalReportHook(PostEvalHook):
    """Post-evaluation hook that generates reports from cached requests and responses."""

    class Params(BaseModel):
        """Configuration parameters for post-evaluation report generation."""

        report_types: List[Literal["html", "json"]] = Field(
            default=["html"],
            description="List of report types to generate (html, json)",
        )
        html_report_size: int | None = Field(
            default=None,
            description="Maximum number of request-response pairs to include in HTML report. If None, includes all available pairs.",
        )

        class Config:
            use_enum_values = True

    def __init__(self, params: Params):
        """
        Initialize the post-evaluation report hook.

        Args:
            params: Configuration parameters
        """
        # Validate report types
        for report_type in params.report_types:
            if report_type not in ["html", "json"]:
                raise ValueError(
                    f"Invalid report type: {report_type}. Must be one of: ['html', 'json']"
                )

        # Store report types as strings for internal use
        self.report_types = params.report_types
        self.html_report_size = params.html_report_size

        # Initialize Jinja2 environment for HTML reports
        if "html" in self.report_types:
            self.env = Environment(
                undefined=StrictUndefined,
                autoescape=select_autoescape(["html", "xml"]),
            )
            self.env.filters["tojson_utf8"] = self._tojson_utf8
            self.template = self.env.from_string(SIMPLE_TEMPLATE)

    def _tojson_utf8(self, data: Any) -> str:
        """Format JSON data for HTML display with UTF-8 support."""
        import html

        return html.escape(json.dumps(data, indent=2, ensure_ascii=False))

    def _get_request_content(self, request_data: Any) -> Any:
        """Extract content from request data."""
        try:
            if isinstance(request_data, bytes):
                request_data = request_data.decode("utf-8")

            if isinstance(request_data, str):
                try:
                    parsed = json.loads(request_data)
                    return parsed  # Return the parsed dict
                except json.JSONDecodeError:
                    return request_data

            # If it's already a dict or other type, return as is
            return request_data
        except Exception:
            return request_data

    def _get_response_content(self, response_data: Any) -> Any:
        """Extract content from response data."""
        try:
            if isinstance(response_data, bytes):
                response_data = response_data.decode("utf-8")

            if isinstance(response_data, str):
                try:
                    parsed = json.loads(response_data)
                    return parsed  # Return the parsed dict
                except json.JSONDecodeError:
                    return response_data

            # If it's already a dict or other type, return as is
            return response_data
        except Exception:
            return str(response_data)

    def _collect_entries(self, cache_dir: Path, api_url: str) -> list:
        """Collect all request-response entries from cache."""
        entries = []

        # Create cache directories if they don't exist
        responses_dir = cache_dir / "responses"
        requests_dir = cache_dir / "requests"
        headers_dir = cache_dir / "headers"

        # Initialize caches with directory paths
        responses_cache = Cache(directory=str(responses_dir))
        requests_cache = Cache(directory=str(requests_dir))
        _ = Cache(directory=str(headers_dir))

        # Get all cache keys from both caches
        response_keys = [key for key in responses_cache.iterkeys()]
        request_keys = [key for key in requests_cache.iterkeys()]

        # Use request keys as primary since they should match response keys
        cache_keys = request_keys if request_keys else response_keys

        get_logger().debug(
            "Cache keys collected for the report",
            types=self.report_types,
            request_keys_len=len(request_keys),
            response_keys_len=len(response_keys),
            cache_keys_len=len(cache_keys),
            dirs=(responses_dir, requests_dir),
        )

        if not cache_keys:
            return []

        # Collect all cache entries
        for cache_key in cache_keys:
            try:
                # Get request data first
                request_content = None
                if cache_key in requests_cache:
                    request_data = requests_cache[cache_key]
                    request_content = self._get_request_content(request_data)
                else:
                    continue

                # Get response data
                response_content = None
                if cache_key in responses_cache:
                    response_data = responses_cache[cache_key]
                    response_content = self._get_response_content(response_data)

                # Add entry data
                entries.append(
                    {
                        "request_data": request_content,
                        "display_request": request_content,  # Already processed by _get_request_content
                        "response": response_content,
                        "endpoint": api_url,
                        "cache_key": cache_key,
                    }
                )
            except Exception:
                continue

        get_logger().debug("Entries collected", num_entries=len(entries))

        # Apply html_report_size limit if specified
        if self.html_report_size is not None and len(entries) > self.html_report_size:
            # Sort by cache key to ensure consistent ordering
            entries.sort(key=lambda x: x["cache_key"])
            entries = entries[: self.html_report_size]
            get_logger().info(
                "Limited HTML report entries",
                total_available=len(cache_keys),
                limited_to=self.html_report_size,
            )

        return entries

    def _generate_html_report(self, entries: list, output_path: Path) -> None:
        """Generate HTML report."""
        html_content = self.template.render(entries=entries)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

    def _generate_json_report(self, entries: list, output_path: Path) -> None:
        """Generate JSON report."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        """Generate reports of cached requests and responses."""
        # Derive cache_dir from output_dir
        cache_dir = Path(context.output_dir) / "cache"

        # Collect entries from cache
        entries = self._collect_entries(cache_dir, context.url)

        if not entries:
            return

        # Generate reports based on configured types
        for report_type in self.report_types:
            if report_type == "html":
                output_path = Path(context.output_dir) / "report.html"
                self._generate_html_report(entries, output_path)
                get_logger().info("Generated HTML report", path=output_path)
            elif report_type == "json":
                output_path = Path(context.output_dir) / "report.json"
                self._generate_json_report(entries, output_path)
                get_logger().info("Generated JSON report", path=output_path)
