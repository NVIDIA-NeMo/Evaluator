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
import re
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

    def _load_tasks_irs(self) -> dict[str, Any] | None:
        """Load tasks-to-harness/container mapping if available."""
        candidates: list[Path] = []
        try:
            repo_root = Path(__file__).resolve().parents[6]
            candidates.append(
                repo_root
                / "packages"
                / "nemo-evaluator-launcher"
                / "src"
                / "nemo_evaluator_launcher"
                / "resources"
                / "all_tasks_irs.yaml"
            )
        except Exception:
            pass

        for candidate in candidates:
            if candidate.exists():
                try:
                    return yaml.safe_load(candidate.read_text(encoding="utf-8"))
                except Exception:
                    return None

        try:
            from importlib import resources as importlib_resources

            with importlib_resources.as_file(
                importlib_resources.files("nemo_evaluator_launcher.resources")
                / "all_tasks_irs.yaml"
            ) as resource_path:
                if resource_path.exists():
                    return yaml.safe_load(resource_path.read_text(encoding="utf-8"))
        except Exception:
            return None

        return None

    def _container_url_from_image(self, image: str) -> str | None:
        """Map an NGC image to a catalog URL when possible."""
        if not image:
            return None
        match = re.match(r"^nvcr\\.io/nvidia/([^/]+)/([^:]+)(?::.+)?$", image)
        if not match:
            return None
        org_or_team, name = match.groups()
        if org_or_team == "eval-factory":
            return (
                "https://catalog.ngc.nvidia.com/orgs/nvidia/teams/"
                f"eval-factory/containers/{name}"
            )
        return f"https://catalog.ngc.nvidia.com/orgs/nvidia/containers/{name}"

    def _resolve_container_info(self, task_name: str | None) -> dict[str, Any]:
        """Resolve benchmark container info from the launcher task registry."""
        if not task_name:
            return {}
        registry = self._load_tasks_irs()
        if not registry:
            return {}

        raw_task = task_name
        harness_hint = None
        if "." in task_name:
            harness_hint, raw_task = task_name.split(".", 1)

        task_entry = None
        for task in registry.get("tasks", []):
            if task.get("name") == raw_task and (
                not harness_hint or task.get("harness") == harness_hint
            ):
                task_entry = task
                break

        if not task_entry:
            for task in registry.get("tasks", []):
                if task.get("name") == raw_task:
                    task_entry = task
                    break

        harness_name = task_entry.get("harness") if task_entry else None
        harness_entry = None
        for harness in registry.get("harnesses", []):
            if harness.get("name") == harness_name:
                harness_entry = harness
                break

        container = harness_entry.get("container") if harness_entry else None
        container_digest = (
            harness_entry.get("container_digest") if harness_entry else None
        )
        container_url = self._container_url_from_image(container) if container else None
        harness_url = harness_entry.get("url") if harness_entry else None

        return {
            "benchmark_task": raw_task,
            "benchmark_harness": harness_name,
            "benchmark_container": container,
            "benchmark_container_digest": container_digest,
            "benchmark_container_url": container_url,
            "benchmark_harness_url": harness_url,
        }

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

    def _normalize_ws(self, value: str) -> str:
        return " ".join(value.split()).strip()

    def _format_value(self, value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _format_stats(self, stats: Any) -> str:
        if not stats:
            return "-"
        if isinstance(stats, dict):
            parts = []
            for key in sorted(stats.keys()):
                val = stats.get(key)
                if isinstance(val, float):
                    parts.append(f"{key}={val:.4f}")
                else:
                    parts.append(f"{key}={val}")
            return ", ".join(parts) if parts else "-"
        return str(stats)

    def _display_metric_path(self, path: str) -> str:
        if ".metrics." in path:
            return path.split(".metrics.", 1)[1]
        return path

    def _collect_scoped_metrics(
        self, results: dict[str, Any], scope_key: str
    ) -> list[dict[str, Any]]:
        scoped = self._get_nested(results, ["results", scope_key])
        rows: list[dict[str, Any]] = []
        if not isinstance(scoped, dict):
            return rows

        for scope_name, scope_data in scoped.items():
            if not isinstance(scope_data, dict):
                continue
            metrics = scope_data.get("metrics")
            if not isinstance(metrics, dict):
                continue
            for row in self._flatten_metrics(metrics, prefix="metrics."):
                path = row.get("path") or ""
                path_display = path[len("metrics.") :] if path.startswith("metrics.") else path
                rows.append(
                    {
                        "scope": scope_name,
                        "path": path,
                        "path_display": path_display,
                        "value": row.get("value"),
                        "value_display": self._format_value(row.get("value")),
                        "stats": row.get("stats"),
                        "stats_display": self._format_stats(row.get("stats")),
                    }
                )
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

    def _extract_user_prompt(self, request_content: Any) -> str:
        if not isinstance(request_content, dict):
            return ""
        if "messages" in request_content:
            messages = request_content.get("messages") or []
            parts: list[str] = []
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    content = self._normalize_content(msg.get("content"))
                    if content:
                        parts.append(content)
            if parts:
                return "\n\n".join(parts)
        if "prompt" in request_content:
            return self._normalize_content(request_content.get("prompt"))
        return ""

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

    def _derive_correctness(self, record: dict[str, Any]) -> Optional[bool]:
        if "symbolic_correct" in record and isinstance(record["symbolic_correct"], bool):
            return record["symbolic_correct"]
        if "correct" in record and isinstance(record["correct"], bool):
            return record["correct"]
        for key in (
            "prompt_level_strict_acc",
            "prompt_level_loose_acc",
            "acc",
            "accuracy",
            "exact_match",
            "is_correct",
        ):
            if key in record:
                value = record[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value > 0
        for key in ("inst_level_strict_acc", "inst_level_loose_acc"):
            if key in record:
                value = record[key]
                if isinstance(value, bool):
                    return value
                if isinstance(value, (int, float)):
                    return value > 0
                if isinstance(value, list) and value:
                    return all(bool(v) for v in value)
        return None

    def _extract_graded_response(self, record: dict[str, Any]) -> str:
        filtered = record.get("filtered_resps")
        if isinstance(filtered, list) and filtered:
            if isinstance(filtered[0], str):
                return filtered[0]
            if isinstance(filtered[0], list) and filtered[0]:
                return str(filtered[0][0])
        resps = record.get("resps")
        if isinstance(resps, list) and resps:
            if isinstance(resps[0], str):
                return resps[0]
            if isinstance(resps[0], list) and resps[0]:
                return str(resps[0][0])
        if isinstance(record.get("generation"), str):
            return record.get("generation")
        serialized = record.get("serialized_output")
        if isinstance(serialized, list) and serialized:
            first = serialized[0]
            if isinstance(first, dict) and isinstance(first.get("content"), str):
                return first.get("content")
        return ""

    def _extract_prompt_variants(self, record: dict[str, Any]) -> list[str]:
        variants: list[str] = []
        doc = record.get("doc")
        if isinstance(doc, dict) and isinstance(doc.get("prompt"), str):
            variants.append(doc.get("prompt"))
        if isinstance(record.get("prompt"), str):
            variants.append(record.get("prompt"))
        problem = record.get("problem")
        options = record.get("options")
        if isinstance(problem, str):
            variants.append(problem)
            if isinstance(options, str):
                variants.append(f"{problem}\n\n{options}")
        arguments = record.get("arguments") or {}
        if isinstance(arguments, dict):
            gen_args = arguments.get("gen_args_0") or {}
            if isinstance(gen_args, dict):
                arg_0 = gen_args.get("arg_0") or []
                if isinstance(arg_0, list) and arg_0:
                    first = arg_0[0]
                    if isinstance(first, str):
                        try:
                            parsed = json.loads(first)
                            if isinstance(parsed, list):
                                for msg in parsed:
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        content = self._normalize_content(msg.get("content"))
                                        if content:
                                            variants.append(content)
                        except Exception:
                            pass
        return [v for v in variants if v]

    def _extract_primary_input(self, record: dict[str, Any]) -> str:
        doc = record.get("doc")
        if isinstance(doc, dict) and isinstance(doc.get("prompt"), str):
            return doc.get("prompt")
        problem = record.get("problem")
        options = record.get("options")
        if isinstance(problem, str):
            if isinstance(options, str):
                return f"{problem}\n\n{options}"
            return problem
        arguments = record.get("arguments") or {}
        if isinstance(arguments, dict):
            gen_args = arguments.get("gen_args_0") or {}
            if isinstance(gen_args, dict):
                arg_0 = gen_args.get("arg_0") or []
                if isinstance(arg_0, list) and arg_0:
                    first = arg_0[0]
                    if isinstance(first, str):
                        try:
                            parsed = json.loads(first)
                            if isinstance(parsed, list):
                                parts: list[str] = []
                                for msg in parsed:
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        content = self._normalize_content(msg.get("content"))
                                        if content:
                                            parts.append(content)
                                if parts:
                                    return "\n\n".join(parts)
                        except Exception:
                            pass
        return ""

    def _extract_target(self, record: dict[str, Any]) -> str:
        def parse_options_string(options_text: str) -> dict[str, str]:
            mapping: dict[str, str] = {}
            for line in options_text.splitlines():
                line = line.strip()
                if not line:
                    continue
                if len(line) >= 2 and line[1] in {")", "."}:
                    label = line[0].upper()
                    value = line[2:].strip()
                    if label.isalpha() and value:
                        mapping[label] = value
            return mapping

        def choice_label(idx: int) -> str:
            if idx < 0:
                return ""
            return chr(ord("A") + idx)

        expected = record.get("expected_answer")
        if expected is not None:
            if isinstance(expected, str):
                expected = expected.strip()
            if expected != "":
                # Try to render expected answer with choices/options when available.
                doc = record.get("doc")
                choices: list[Any] | None = None
                options_map: dict[str, str] = {}
                if isinstance(doc, dict):
                    maybe_choices = doc.get("choices") or doc.get("options")
                    if isinstance(maybe_choices, list):
                        choices = maybe_choices
                options = record.get("options")
                if isinstance(options, str):
                    options_map = parse_options_string(options)
                if isinstance(expected, str) and len(expected) == 1 and expected.isalpha():
                    letter = expected.upper()
                    if letter in options_map:
                        return f"{letter}) {options_map[letter]}"
                if isinstance(expected, (int, float)) and choices:
                    idx = int(expected)
                    if 0 <= idx < len(choices):
                        letter = choice_label(idx)
                        return f"{letter}) {choices[idx]}"
                if isinstance(expected, str) and choices and expected in choices:
                    return expected
                return str(expected)

        target = record.get("target")
        doc = record.get("doc")
        choices: list[Any] | None = None
        options_map: dict[str, str] = {}
        if isinstance(doc, dict):
            maybe_choices = doc.get("choices") or doc.get("options")
            if isinstance(maybe_choices, list):
                choices = maybe_choices
        options = record.get("options")
        if isinstance(options, str):
            options_map = parse_options_string(options)
        expected_answer = record.get("expected_answer")
        if isinstance(expected_answer, str):
            expected_letter = expected_answer.strip().upper()
            if expected_letter in options_map:
                return f"{expected_letter}) {options_map[expected_letter]}"
            if expected_answer.strip() and expected_answer.strip() in options_map:
                letter = expected_answer.strip()
                return f"{letter}) {options_map[letter]}"
            if expected_answer.strip():
                return expected_answer.strip()

        # Direct string targets that are not numeric
        if isinstance(target, str) and target and not target.isdigit():
            return target

        # Map numeric targets to choices when available
        target_idx: int | None = None
        if isinstance(target, int):
            target_idx = target
        elif isinstance(target, str) and target.isdigit():
            target_idx = int(target)

        if target_idx is not None and choices and 0 <= target_idx < len(choices):
            letter = choice_label(target_idx)
            return f"{letter}) {choices[target_idx]}"

        # Map doc-level answer/label when provided
        if isinstance(doc, dict):
            doc_answer = doc.get("answer") or doc.get("label") or doc.get("gold")
            if isinstance(doc_answer, int) and choices and 0 <= doc_answer < len(choices):
                letter = choice_label(doc_answer)
                return f"{letter}) {choices[doc_answer]}"
            if isinstance(doc_answer, str):
                stripped = doc_answer.strip()
                if choices:
                    if stripped.isdigit():
                        idx = int(stripped)
                        if 0 <= idx < len(choices):
                            letter = choice_label(idx)
                            return f"{letter}) {choices[idx]}"
                    if len(stripped) == 1 and stripped.isalpha():
                        idx = ord(stripped.upper()) - ord("A")
                        if 0 <= idx < len(choices):
                            letter = choice_label(idx)
                            return f"{letter}) {choices[idx]}"
                    if stripped in choices:
                        return stripped
                if stripped and len(stripped) == 1 and stripped.isalpha():
                    letter = stripped.upper()
                    if letter in options_map:
                        return f"{letter}) {options_map[letter]}"
                if stripped:
                    return stripped

            doc_target = doc.get("target")
            if isinstance(doc_target, str) and doc_target:
                return doc_target

            # Instruction-following tasks may not have a single ground truth.
            if "instruction_id_list" in doc:
                return "N/A (instruction-following)"

        # Fallback to numeric targets when nothing else is available
        if target is not None:
            return str(target)
        return ""

    def _collect_grading_records(self, output_dir: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for path in output_dir.rglob("*.jsonl"):
            name = path.name
            if not (name.startswith("samples_") or name == "output.jsonl"):
                continue
            try:
                with path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            record = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(record, dict):
                            continue
                        variants = self._extract_prompt_variants(record)
                        if not variants:
                            continue
                        graded_response = self._extract_graded_response(record)
                        correctness = self._derive_correctness(record)
                        target_display = self._extract_target(record)
                        records.append(
                            {
                                "variants": variants,
                                "input": self._extract_primary_input(record),
                                "problem": record.get("problem")
                                if isinstance(record.get("problem"), str)
                                else None,
                                "options": record.get("options")
                                if isinstance(record.get("options"), str)
                                else None,
                                "target": target_display,
                                "target_display": target_display,
                                "graded_response": graded_response,
                                "correct": correctness,
                                "expected": record.get("expected_answer"),
                                "predicted": record.get("predicted_answer"),
                                "score": 1 if correctness is True else 0 if correctness is False else None,
                                "label": (
                                    "correct"
                                    if correctness is True
                                    else "incorrect"
                                    if correctness is False
                                    else "unknown"
                                ),
                                "metrics": {
                                    k: record.get(k)
                                    for k in [
                                        "prompt_level_strict_acc",
                                        "prompt_level_loose_acc",
                                        "inst_level_strict_acc",
                                        "inst_level_loose_acc",
                                        "symbolic_correct",
                                    ]
                                    if k in record
                                },
                                "source": str(path),
                            }
                        )
            except Exception:
                continue
        return records

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

        grading_records = self._collect_grading_records(output_dir)
        grading_exact: dict[str, dict[str, Any]] = {}
        grading_exact_norm: dict[str, dict[str, Any]] = {}
        for record in grading_records:
            for variant in record.get("variants", []):
                if not variant:
                    continue
                if variant not in grading_exact:
                    grading_exact[variant] = record
                normalized_variant = self._normalize_ws(variant)
                if normalized_variant and normalized_variant not in grading_exact_norm:
                    grading_exact_norm[normalized_variant] = record

        def find_grading(prompt: str) -> Optional[dict[str, Any]]:
            if not prompt:
                return None
            if prompt in grading_exact:
                return grading_exact[prompt]
            normalized_prompt = self._normalize_ws(prompt)
            if normalized_prompt in grading_exact_norm:
                return grading_exact_norm[normalized_prompt]

            best_match: Optional[dict[str, Any]] = None
            best_score = 0
            best_len = 0
            for record in grading_records:
                score = 0
                record_len = 0
                problem = record.get("problem") or record.get("input")
                if isinstance(problem, str):
                    normalized_problem = self._normalize_ws(problem)
                    if normalized_problem and normalized_problem in normalized_prompt:
                        score += 3
                        record_len = max(record_len, len(normalized_problem))
                options = record.get("options")
                if isinstance(options, str):
                    normalized_options = self._normalize_ws(options)
                    if normalized_options and normalized_options in normalized_prompt:
                        score += 3
                        record_len = max(record_len, len(normalized_options))

                for variant in record.get("variants", []):
                    if not variant:
                        continue
                    normalized_variant = self._normalize_ws(variant)
                    if not normalized_variant:
                        continue
                    if (
                        normalized_variant in normalized_prompt
                        or normalized_prompt in normalized_variant
                    ):
                        score += 1
                        record_len = max(record_len, len(normalized_variant))

                if score > best_score or (
                    score == best_score and record_len > best_len
                ):
                    best_match = record
                    best_score = score
                    best_len = record_len

            if best_score == 0:
                return None
            return best_match

        for entry in entries:
            entry.setdefault("graded_label", "unknown")
            entry.setdefault("graded_response", None)
            entry.setdefault("graded_input", None)
            entry.setdefault("graded_target", None)
            entry.setdefault("target_display", None)
            entry.setdefault("prompt_sections", [])
            entry.setdefault("graded_expected", None)
            entry.setdefault("graded_predicted", None)
            entry.setdefault("graded_score", None)
            entry.setdefault("graded_metrics", None)
            entry.setdefault("graded_source", None)
            prompt = self._extract_user_prompt(entry.get("request_data"))
            match = find_grading(prompt)
            if match:
                entry["graded_response"] = match.get("graded_response")
                entry["graded_expected"] = match.get("expected")
                entry["graded_predicted"] = match.get("predicted")
                entry["graded_correct"] = match.get("correct")
                entry["graded_input"] = match.get("input")
                entry["graded_target"] = match.get("target")
                entry["target_display"] = match.get("target_display") or match.get("target")
                entry["graded_score"] = match.get("score")
                entry["graded_metrics"] = match.get("metrics")
                entry["graded_source"] = match.get("source")
                entry["graded_label"] = match.get("label") or "unknown"
                graded_blob = self._safe_json_dumps(match.get("graded_response"))
                graded_input = self._safe_json_dumps(match.get("input"))
                graded_target = self._safe_json_dumps(match.get("target"))
                entry["search_blob"] = (
                    f"{entry.get('search_blob', '')} {graded_blob} {graded_input} {graded_target}".lower()
                )

            prompt_sections: list[dict[str, str]] = []
            seen_prompts: set[str] = set()
            graded_input = entry.get("graded_input")
            prompt_preview = entry.get("prompt_preview")

            def add_prompt(label: str, text: str | None) -> None:
                if not text:
                    return
                normalized = self._normalize_ws(text)
                if not normalized or normalized in seen_prompts:
                    return
                seen_prompts.add(normalized)
                prompt_sections.append({"label": label, "text": text})

            add_prompt("Prompt", graded_input or prompt_preview)
            entry["prompt_sections"] = prompt_sections

        for entry in entries:
            if entry.get("request_chars") is None:
                entry["request_chars"] = len(
                    self._safe_json_dumps(entry.get("request_data"))
                )
            if entry.get("response_chars") is None:
                entry["response_chars"] = len(
                    self._safe_json_dumps(entry.get("response"))
                )

        error_count = sum(1 for entry in entries if entry.get("has_error"))
        tool_count = sum(1 for entry in entries if entry.get("has_tool_calls"))
        correct_count = sum(
            1 for entry in entries if (entry.get("graded_label") or "unknown") == "correct"
        )
        incorrect_count = sum(
            1
            for entry in entries
            if (entry.get("graded_label") or "unknown") == "incorrect"
        )
        unknown_count = sum(
            1 for entry in entries if (entry.get("graded_label") or "unknown") == "unknown"
        )
        graded_count = correct_count + incorrect_count
        report_count = len(entries)
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
        if not task_name:
            task_name = self._get_nested(results, ["config", "params", "task"])
        if not task_name:
            task_name = self._get_nested(results, ["config", "type"])
        container_info = self._resolve_container_info(task_name)

        if task_name:
            summary_items.append({"label": "Task", "value": str(task_name)})
        if container_info.get("benchmark_container"):
            summary_items.append(
                {"label": "Benchmark Container", "value": container_info["benchmark_container"]}
            )
        if container_info.get("benchmark_harness"):
            summary_items.append(
                {"label": "Harness", "value": container_info["benchmark_harness"]}
            )

        command = self._get_nested(results, ["command"])
        if command:
            summary_items.append({"label": "Command", "value": str(command)})
        if model_set:
            summary_items.append({"label": "Models", "value": ", ".join(model_set)})

        rollup_rows = self._collect_scoped_metrics(results, "groups")
        task_rows = self._collect_scoped_metrics(results, "tasks")

        metrics_rows: list[dict[str, Any]] = []
        results_metrics = self._get_nested(results, ["results"])
        if isinstance(results_metrics, dict):
            raw_metrics = self._flatten_metrics(results_metrics)
            metrics_rows = [
                {
                    "path": row.get("path"),
                    "path_display": self._display_metric_path(row.get("path", "")),
                    "value": row.get("value"),
                    "value_display": self._format_value(row.get("value")),
                    "stats": row.get("stats"),
                    "stats_display": self._format_stats(row.get("stats")),
                }
                for row in raw_metrics
            ]

        perf_rows: list[dict[str, Any]] = []
        if isinstance(eval_metrics, dict):
            raw_perf = self._flatten_numeric(eval_metrics)
            perf_rows = [
                {
                    "path": row.get("path"),
                    "path_display": row.get("path"),
                    "value": row.get("value"),
                    "value_display": self._format_value(row.get("value")),
                }
                for row in raw_perf
            ]

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
            "model_id": model_id,
            "task_name": task_name,
            "benchmark_container": container_info.get("benchmark_container"),
            "benchmark_container_url": container_info.get("benchmark_container_url"),
            "benchmark_harness": container_info.get("benchmark_harness"),
            "benchmark_harness_url": container_info.get("benchmark_harness_url"),
            "summary_items": summary_items,
            "rollup_rows": rollup_rows,
            "task_rows": task_rows,
            "metrics_rows": metrics_rows,
            "perf_rows": perf_rows,
            "artifacts_rows": artifacts_rows,
            "raw_results": raw_results,
            "raw_run_config": raw_run_config,
            "raw_eval_metrics": raw_eval_metrics,
            "stats": {
                "error_count": error_count,
                "tool_count": tool_count,
                "correct_count": correct_count,
                "incorrect_count": incorrect_count,
                "unknown_count": unknown_count,
                "graded_count": graded_count,
                "report_count": report_count,
                "accuracy": (
                    (correct_count / (correct_count + incorrect_count))
                    if (correct_count + incorrect_count)
                    else 0
                ),
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
                "grading": ["correct", "incorrect", "unknown"],
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
