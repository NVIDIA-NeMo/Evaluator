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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, List, Literal, Optional

from jinja2 import Environment, StrictUndefined, select_autoescape
from pydantic import BaseModel, Field
import yaml

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

    def _collect_entries(self, cache_dir: Path, api_url: str) -> tuple[list, int]:
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
            return [], 0

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
                request_blob = self._safe_json_dumps(request_content)
                response_blob = self._safe_json_dumps(response_content)
                search_blob = f"{cache_key} {request_blob} {response_blob}".lower()

                model = (
                    request_content.get("model")
                    if isinstance(request_content, dict)
                    else None
                )
                request_type = self._detect_request_type(request_content)
                prompt_preview = self._extract_prompt_preview(request_content)
                finish_reasons = self._extract_finish_reasons(response_content)
                usage = self._extract_usage(response_content)
                has_error = self._has_error(response_content)
                has_tool_calls = self._has_tool_calls(response_content)

                entries.append(
                    {
                        "request_data": request_content,
                        "display_request": request_content,  # Already processed by _get_request_content
                        "response": response_content,
                        "endpoint": api_url,
                        "cache_key": cache_key,
                        "search_blob": search_blob,
                        "model": model or "unknown",
                        "request_type": request_type,
                        "prompt_preview": prompt_preview,
                        "finish_reasons": finish_reasons,
                        "usage": usage,
                        "has_error": has_error,
                        "has_tool_calls": has_tool_calls,
                        "request_chars": len(request_blob),
                        "response_chars": len(response_blob),
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

        return entries, len(cache_keys)

    def _load_json(self, path: Path) -> Any:
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}

    def _get_nested(self, data: dict[str, Any], keys: Iterable[str]) -> Any:
        current: Any = data
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    def _flatten_metrics(self, data: Any, prefix: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            if "value" in data and isinstance(data["value"], (int, float, str)):
                rows.append(
                    {
                        "path": prefix.rstrip("."),
                        "value": data.get("value"),
                        "stats": data.get("stats"),
                    }
                )
            for key, value in data.items():
                next_prefix = f"{prefix}{key}."
                rows.extend(self._flatten_metrics(value, next_prefix))
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                next_prefix = f"{prefix}{idx}."
                rows.extend(self._flatten_metrics(value, next_prefix))
        return rows

    def _flatten_numeric(self, data: Any, prefix: str = "") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        if isinstance(data, dict):
            for key, value in data.items():
                next_prefix = f"{prefix}{key}."
                if isinstance(value, (int, float)):
                    rows.append({"path": next_prefix.rstrip("."), "value": value})
                else:
                    rows.extend(self._flatten_numeric(value, next_prefix))
        elif isinstance(data, list):
            for idx, value in enumerate(data):
                next_prefix = f"{prefix}{idx}."
                rows.extend(self._flatten_numeric(value, next_prefix))
        return rows

    def _safe_json_dumps(self, data: Any) -> str:
        try:
            return json.dumps(data, ensure_ascii=False)
        except Exception:
            return str(data)

    def _detect_request_type(self, request_content: Any) -> str:
        if isinstance(request_content, dict):
            if "messages" in request_content:
                return "chat"
            if "prompt" in request_content:
                return "completion"
        return "unknown"

    def _normalize_content(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        parts.append(str(item.get("text", "")))
            return " ".join(p for p in parts if p).strip()
        if isinstance(content, dict) and "text" in content:
            return str(content.get("text", ""))
        return str(content)

    def _extract_prompt_preview(self, request_content: Any) -> str:
        if not isinstance(request_content, dict):
            return ""
        if "messages" in request_content:
            messages = request_content.get("messages") or []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = self._normalize_content(msg.get("content"))
                    if content:
                        return content[:160]
        if "prompt" in request_content:
            prompt = self._normalize_content(request_content.get("prompt"))
            return prompt[:160]
        return ""

    def _extract_finish_reasons(self, response_content: Any) -> list[str]:
        reasons: list[str] = []
        if isinstance(response_content, dict):
            choices = response_content.get("choices") or []
            for choice in choices:
                if isinstance(choice, dict):
                    reason = choice.get("finish_reason")
                    if reason:
                        reasons.append(str(reason))
        return reasons

    def _extract_usage(self, response_content: Any) -> Optional[dict[str, Any]]:
        if isinstance(response_content, dict):
            usage = response_content.get("usage")
            if isinstance(usage, dict):
                return {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                }
        return None

    def _has_error(self, response_content: Any) -> bool:
        if isinstance(response_content, dict):
            if "error" in response_content or "errors" in response_content:
                return True
        return False

    def _has_tool_calls(self, response_content: Any) -> bool:
        if not isinstance(response_content, dict):
            return False
        choices = response_content.get("choices") or []
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if choice.get("tool_calls") or choice.get("function_call"):
                return True
            message = choice.get("message")
            if isinstance(message, dict):
                if message.get("tool_calls") or message.get("function_call"):
                    return True
        return False

    def _format_bytes(self, size: int) -> str:
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.0f} {unit}"
            size = size / 1024
        return f"{size:.0f} TB"

    def _summarize_artifacts(self, output_dir: Path) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        if not output_dir.exists():
            return rows
        for child in sorted(output_dir.iterdir()):
            if child.is_file():
                size = child.stat().st_size
                rows.append(
                    {
                        "name": child.name,
                        "detail": self._format_bytes(size),
                    }
                )
            elif child.is_dir():
                try:
                    count = sum(1 for _ in child.iterdir())
                except Exception:
                    count = 0
                rows.append(
                    {
                        "name": f"{child.name}/",
                        "detail": f"{count} items",
                    }
                )
        return rows

    def _generate_html_report(
        self, entries: list, output_path: Path, meta: dict[str, Any]
    ) -> None:
        """Generate HTML report."""
        html_content = self.template.render(entries=entries, meta=meta)
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
        output_dir = Path(context.output_dir)

        # Collect entries from cache
        entries, total_available = self._collect_entries(cache_dir, context.url)

        if not entries:
            return

        # Generate reports based on configured types
        # Load additional artifacts for richer HTML report.
        results_path = output_dir / "results.yml"
        run_config_path = output_dir / "run_config.yml"
        eval_metrics_path = output_dir / "eval_factory_metrics.json"

        results = self._load_yaml(results_path)
        run_config = self._load_yaml(run_config_path)
        eval_metrics = self._load_json(eval_metrics_path) or {}

        error_count = sum(1 for entry in entries if entry.get("has_error"))
        tool_count = sum(1 for entry in entries if entry.get("has_tool_calls"))
        avg_request_chars = (
            sum(entry.get("request_chars", 0) for entry in entries) / len(entries)
        )
        avg_response_chars = (
            sum(entry.get("response_chars", 0) for entry in entries) / len(entries)
        )
        model_set = sorted(
            {entry.get("model") for entry in entries if entry.get("model")}
        )
        request_type_counts: dict[str, int] = {}
        finish_reason_counts: dict[str, int] = {}
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for entry in entries:
            request_type = entry.get("request_type") or "unknown"
            request_type_counts[request_type] = request_type_counts.get(request_type, 0) + 1
            for reason in entry.get("finish_reasons") or []:
                finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1
            usage = entry.get("usage") or {}
            for key in usage_totals:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    usage_totals[key] += value

        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

        summary_items: list[dict[str, str]] = []
        summary_items.append({"label": "Endpoint", "value": context.url})
        summary_items.append({"label": "Generated At (UTC)", "value": generated_at})
        summary_items.append({"label": "Shown Entries", "value": str(len(entries))})
        summary_items.append({"label": "Total Cached", "value": str(total_available)})
        summary_items.append({"label": "Output Dir", "value": str(output_dir)})

        git_hash = self._get_nested(results, ["git_hash"])
        if git_hash:
            summary_items.append({"label": "Git Hash", "value": str(git_hash)})

        model_id = self._get_nested(results, ["target", "api_endpoint", "model_id"])
        if not model_id:
            model_id = self._get_nested(run_config, ["target", "api_endpoint", "model_id"])
        if model_id:
            summary_items.append({"label": "Model ID", "value": str(model_id)})

        task_name = self._get_nested(run_config, ["task", "name"])
        if not task_name:
            task_name = self._get_nested(run_config, ["task", "task_name"])
        if task_name:
            summary_items.append({"label": "Task", "value": str(task_name)})

        command = self._get_nested(results, ["command"])
        if command:
            summary_items.append({"label": "Command", "value": str(command)})
        if model_set:
            summary_items.append({"label": "Models", "value": ", ".join(model_set)})

        metrics_rows: list[dict[str, Any]] = []
        results_metrics = self._get_nested(results, ["results"])
        if isinstance(results_metrics, dict):
            metrics_rows = self._flatten_metrics(results_metrics)

        perf_rows: list[dict[str, Any]] = []
        if isinstance(eval_metrics, dict):
            perf_rows = self._flatten_numeric(eval_metrics)

        artifacts_rows = self._summarize_artifacts(output_dir)

        raw_results = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
        raw_run_config = (
            run_config_path.read_text(encoding="utf-8") if run_config_path.exists() else ""
        )
        raw_eval_metrics = ""
        if eval_metrics_path.exists():
            try:
                raw_eval_metrics = json.dumps(eval_metrics, indent=2, ensure_ascii=False)
            except Exception:
                raw_eval_metrics = eval_metrics_path.read_text(encoding="utf-8")

        meta = {
            "endpoint": context.url,
            "generated_at": generated_at,
            "shown_entries": len(entries),
            "total_entries": total_available,
            "limited": total_available > len(entries),
            "summary_items": summary_items,
            "metrics_rows": metrics_rows,
            "perf_rows": perf_rows,
            "artifacts_rows": artifacts_rows,
            "raw_results": raw_results,
            "raw_run_config": raw_run_config,
            "raw_eval_metrics": raw_eval_metrics,
            "stats": {
                "error_count": error_count,
                "tool_count": tool_count,
                "error_rate": (error_count / len(entries)) if entries else 0,
                "avg_request_chars": avg_request_chars,
                "avg_response_chars": avg_response_chars,
                "usage_totals": usage_totals,
                "request_type_counts": request_type_counts,
                "finish_reason_counts": finish_reason_counts,
            },
            "filters": {
                "models": model_set,
                "request_types": sorted(request_type_counts.keys()),
                "finish_reasons": sorted(finish_reason_counts.keys()),
            },
        }

        for report_type in self.report_types:
            if report_type == "html":
                output_path = Path(context.output_dir) / "report.html"
                self._generate_html_report(entries, output_path, meta)
                get_logger().info("Generated HTML report", path=output_path)
            elif report_type == "json":
                output_path = Path(context.output_dir) / "report.json"
                self._generate_json_report(entries, output_path)
                get_logger().info("Generated JSON report", path=output_path)
