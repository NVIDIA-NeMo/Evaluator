# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Analyze evaluation artifacts and generate a rich HTML report."""

from __future__ import annotations

from dataclasses import dataclass
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from simple_parsing import field


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        import json

        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        from omegaconf import OmegaConf

        data = OmegaConf.load(path)
        return OmegaConf.to_container(data, resolve=True) or {}
    except Exception:
        return {}


def _get_nested(data: dict[str, Any], keys: Iterable[str]) -> Any:
    current: Any = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _flatten_metrics(data: Any, prefix: str = "") -> list[dict[str, Any]]:
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
            rows.extend(_flatten_metrics(value, next_prefix))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}{idx}."
            rows.extend(_flatten_metrics(value, next_prefix))
    return rows


def _flatten_numeric(data: Any, prefix: str = "") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(data, dict):
        for key, value in data.items():
            next_prefix = f"{prefix}{key}."
            if isinstance(value, (int, float)):
                rows.append({"path": next_prefix.rstrip("."), "value": value})
            else:
                rows.extend(_flatten_numeric(value, next_prefix))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            next_prefix = f"{prefix}{idx}."
            rows.extend(_flatten_numeric(value, next_prefix))
    return rows


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def _format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.0f} {unit}"
        size = size / 1024
    return f"{size:.0f} TB"

def _safe_json_dumps(data: Any) -> str:
    try:
        import json

        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return str(data)


def _detect_request_type(request_content: Any) -> str:
    if isinstance(request_content, dict):
        if "messages" in request_content:
            return "chat"
        if "prompt" in request_content:
            return "completion"
    return "unknown"


def _extract_finish_reasons(response_content: Any) -> list[str]:
    reasons: list[str] = []
    if isinstance(response_content, dict):
        choices = response_content.get("choices") or []
        for choice in choices:
            if isinstance(choice, dict):
                reason = choice.get("finish_reason")
                if reason:
                    reasons.append(str(reason))
    return reasons


def _extract_usage_from_entry(entry: dict[str, Any]) -> dict[str, Any]:
    usage = entry.get("usage")
    if isinstance(usage, dict):
        return usage
    response = entry.get("response")
    if isinstance(response, dict):
        usage = response.get("usage")
        if isinstance(usage, dict):
            return usage
    return {}


def _has_error(entry: dict[str, Any]) -> bool:
    if entry.get("has_error"):
        return True
    response = entry.get("response")
    if isinstance(response, dict):
        return "error" in response or "errors" in response
    return False


def _has_tool_calls(entry: dict[str, Any]) -> bool:
    if entry.get("has_tool_calls"):
        return True
    response = entry.get("response")
    if not isinstance(response, dict):
        return False
    choices = response.get("choices") or []
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


def _render_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "<div class=\"empty\">No data available.</div>"
    header_html = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body_html = ""
    for row in rows:
        cells = "".join(f"<td>{escape(cell)}</td>" for cell in row)
        body_html += f"<tr>{cells}</tr>"
    return f"""
    <table>
        <thead><tr>{header_html}</tr></thead>
        <tbody>{body_html}</tbody>
    </table>
    """


def _render_key_value(items: list[tuple[str, str]]) -> str:
    if not items:
        return "<div class=\"empty\">No summary details available.</div>"
    rows = "".join(
        f"<div class=\"kv\"><span>{escape(k)}</span><strong>{escape(v)}</strong></div>"
        for k, v in items
    )
    return f"<div class=\"kv-grid\">{rows}</div>"


def _render_samples(samples: list[dict[str, Any]]) -> str:
    if not samples:
        return "<div class=\"empty\">No samples found in report.json.</div>"
    def _pretty(value: Any) -> str:
        try:
            import json

            if isinstance(value, (dict, list)):
                return json.dumps(value, indent=2, ensure_ascii=False)
        except Exception:
            pass
        return _format_value(value)

    entries_html = ""
    for idx, entry in enumerate(samples, start=1):
        cache_key = escape(str(entry.get("cache_key", "-")))
        request = escape(_pretty(entry.get("request_data")))
        response = escape(_pretty(entry.get("response")))
        graded_response = escape(_pretty(entry.get("graded_response"))) if entry.get("graded_response") else ""
        graded_label = entry.get("graded_label") or "unknown"
        graded_expected = escape(str(entry.get("graded_expected"))) if entry.get("graded_expected") else ""
        graded_predicted = escape(str(entry.get("graded_predicted"))) if entry.get("graded_predicted") else ""
        badge_class = {
            "correct": "badge-correct",
            "incorrect": "badge-incorrect",
            "unknown": "badge-unknown",
        }.get(graded_label, "badge-unknown")
        entries_html += f"""
        <div class="entry">
            <div class="entry-head">
                <span class="pill">Sample {idx}</span>
                <span class="muted">Cache Key: <strong>{cache_key}</strong></span>
                <span class="{badge_class}">{escape(graded_label)}</span>
                {f'<span class=\"muted\">Expected: <strong>{graded_expected}</strong></span>' if graded_expected else ''}
                {f'<span class=\"muted\">Predicted: <strong>{graded_predicted}</strong></span>' if graded_predicted else ''}
            </div>
            <div class="block">
                <div class="block-title">Request</div>
                <pre>{request}</pre>
            </div>
            <div class="block">
                <div class="block-title">Response</div>
                <pre>{response}</pre>
            </div>
            {f'''<div class="block"><div class="block-title">Graded Response</div><pre>{graded_response}</pre></div>''' if graded_response else ''}
        </div>
        """
    return entries_html


def render_analysis_html(payload: dict[str, Any]) -> str:
    summary_html = _render_key_value(payload.get("summary", []))
    stats_html = _render_key_value(payload.get("stats", []))
    metrics_table = _render_table(
        ["Metric Path", "Value", "Stats"],
        payload.get("metrics_rows", []),
    )
    perf_table = _render_table(
        ["Metric Path", "Value"],
        payload.get("perf_rows", []),
    )
    artifacts_table = _render_table(
        ["Artifact", "Size"],
        payload.get("artifact_rows", []),
    )
    finish_table = _render_table(
        ["Finish Reason", "Count"],
        payload.get("finish_rows", []),
    )
    samples_html = _render_samples(payload.get("samples", []))
    raw_results = payload.get("raw_results", "")
    raw_run_config = payload.get("raw_run_config", "")
    raw_eval_metrics = payload.get("raw_eval_metrics", "")

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Evaluation Analysis</title>
  <style>
    :root {{
      --bg: #0f1116;
      --panel: #141824;
      --panel-2: #10131b;
      --text: #e8ecf1;
      --muted: #9aa6b2;
      --accent: #6f9bff;
      --accent-2: #60d394;
      --border: rgba(255, 255, 255, 0.08);
      --shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }}
    body {{
      font-family: "Space Grotesk", "IBM Plex Sans", "Source Sans 3", sans-serif;
      margin: 0;
      padding: 30px;
      color: var(--text);
      background: radial-gradient(1200px 700px at 15% -20%, #1e2a3b 0%, rgba(15, 17, 22, 0) 65%),
                  radial-gradient(900px 900px at 110% 10%, #1f2440 0%, rgba(15, 17, 22, 0) 60%),
                  var(--bg);
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 2.2rem;
    }}
    .subtitle {{
      color: var(--muted);
      margin-bottom: 22px;
    }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 20px;
      margin-bottom: 18px;
      box-shadow: var(--shadow);
    }}
    .section h2 {{
      margin: 0 0 12px;
      font-size: 1.2rem;
    }}
    .kv-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .kv {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }}
    .kv span {{
      color: var(--muted);
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .kv strong {{
      font-size: 1rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.9rem;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      word-break: break-word;
    }}
    th {{
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .empty {{
      color: var(--muted);
      padding: 18px;
      border: 1px dashed var(--border);
      border-radius: 10px;
      text-align: center;
    }}
    .entry {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin-bottom: 12px;
    }}
    .entry-head {{
      display: flex;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }}
    .pill {{
      background: rgba(111, 155, 255, 0.2);
      color: var(--accent);
      border: 1px solid rgba(111, 155, 255, 0.35);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.78rem;
    }}
    .badge-correct {{
      background: rgba(126, 224, 129, 0.16);
      color: #7ee081;
      border: 1px solid rgba(126, 224, 129, 0.35);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
    }}
    .badge-incorrect {{
      background: rgba(255, 154, 154, 0.16);
      color: #ff9a9a;
      border: 1px solid rgba(255, 154, 154, 0.35);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
    }}
    .badge-unknown {{
      background: rgba(199, 199, 199, 0.12);
      color: #c7c7c7;
      border: 1px solid rgba(199, 199, 199, 0.3);
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.75rem;
    }}
    .muted {{
      color: var(--muted);
      font-size: 0.86rem;
    }}
    .block {{
      background: #0c0f16;
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      margin-bottom: 8px;
    }}
    details {{
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 10px 12px;
      margin-top: 12px;
      background: var(--panel-2);
    }}
    details summary {{
      cursor: pointer;
      color: var(--accent-2);
      font-weight: 600;
    }}
    .block-title {{
      color: var(--muted);
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      margin-bottom: 6px;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font-family: "IBM Plex Mono", "SFMono-Regular", ui-monospace, monospace;
      font-size: 0.82rem;
      color: #d9e2ef;
    }}
    @media (max-width: 720px) {{
      body {{
        padding: 20px;
      }}
      h1 {{
        font-size: 1.8rem;
      }}
    }}
  </style>
</head>
<body>
  <h1>Evaluation Analysis</h1>
  <div class="subtitle">Post-run insights from evaluation artifacts.</div>

  <div class="section">
    <h2>Run Summary</h2>
    {summary_html}
  </div>

  <div class="section">
    <h2>Sample Stats</h2>
    {stats_html}
  </div>

  <div class="section">
    <h2>Score Metrics</h2>
    {metrics_table}
  </div>

  <div class="section">
    <h2>Performance Metrics</h2>
    {perf_table}
  </div>

  <div class="section">
    <h2>Artifacts</h2>
    {artifacts_table}
    {f'''<details><summary>Raw results.yml</summary><pre>{escape(raw_results)}</pre></details>''' if raw_results else ''}
    {f'''<details><summary>Raw run_config.yml</summary><pre>{escape(raw_run_config)}</pre></details>''' if raw_run_config else ''}
    {f'''<details><summary>Raw eval_factory_metrics.json</summary><pre>{escape(raw_eval_metrics)}</pre></details>''' if raw_eval_metrics else ''}
  </div>

  <div class="section">
    <h2>Finish Reasons</h2>
    {finish_table}
  </div>

  <div class="section">
    <h2>Sample Interactions</h2>
    {samples_html}
  </div>
</body>
</html>
"""


@dataclass
class Cmd:
    """Generate an HTML analysis report from a task artifacts directory."""

    artifacts_dir: str = field(
        positional=True,
        help="Path to a task artifacts directory (contains results.yml, report.json, eval_factory_metrics.json).",
    )
    output: str | None = field(
        default=None,
        alias=["--output", "-o"],
        help="Output HTML path (default: <artifacts_dir>/analysis.html).",
    )
    sample_limit: int = field(
        default=20,
        alias=["--sample-limit"],
        help="Maximum number of sample interactions to include (default: 20).",
    )

    def execute(self) -> None:
        artifacts_dir = Path(self.artifacts_dir).expanduser()
        if not artifacts_dir.exists():
            print(f"Error: artifacts_dir not found: {artifacts_dir}")
            return

        output_path = (
            Path(self.output).expanduser()
            if self.output
            else artifacts_dir / "analysis.html"
        )

        results_path = artifacts_dir / "results.yml"
        run_config_path = artifacts_dir / "run_config.yml"
        metrics_path = artifacts_dir / "eval_factory_metrics.json"
        report_path = artifacts_dir / "report.json"

        results = _load_yaml(results_path)
        run_config = _load_yaml(run_config_path)
        eval_metrics = _load_json(metrics_path) or {}
        report_entries = _load_json(report_path) or []

        raw_results = results_path.read_text(encoding="utf-8") if results_path.exists() else ""
        raw_run_config = (
            run_config_path.read_text(encoding="utf-8") if run_config_path.exists() else ""
        )
        raw_eval_metrics = ""
        if metrics_path.exists():
            try:
                raw_eval_metrics = json.dumps(eval_metrics, indent=2, ensure_ascii=False)
            except Exception:
                raw_eval_metrics = metrics_path.read_text(encoding="utf-8")

        if not isinstance(report_entries, list):
            report_entries = []

        sample_count = len(report_entries)
        error_count = sum(1 for entry in report_entries if _has_error(entry))
        tool_count = sum(1 for entry in report_entries if _has_tool_calls(entry))
        correct_count = sum(
            1 for entry in report_entries if entry.get("graded_label") == "correct"
        )
        incorrect_count = sum(
            1 for entry in report_entries if entry.get("graded_label") == "incorrect"
        )
        unknown_count = sum(
            1 for entry in report_entries if entry.get("graded_label") == "unknown"
        )
        model_set = sorted(
            {entry.get("model") for entry in report_entries if entry.get("model")}
        )
        request_type_counts: dict[str, int] = {}
        finish_reason_counts: dict[str, int] = {}
        usage_totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        request_char_sum = 0
        response_char_sum = 0

        for entry in report_entries:
            request_type = entry.get("request_type") or _detect_request_type(
                entry.get("request_data")
            )
            request_type_counts[request_type] = request_type_counts.get(request_type, 0) + 1
            for reason in entry.get("finish_reasons") or _extract_finish_reasons(
                entry.get("response")
            ):
                finish_reason_counts[reason] = finish_reason_counts.get(reason, 0) + 1

            usage = _extract_usage_from_entry(entry)
            for key in usage_totals:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    usage_totals[key] += value

            req_chars = entry.get("request_chars")
            if not isinstance(req_chars, int):
                req_chars = len(_safe_json_dumps(entry.get("request_data")))
            request_char_sum += req_chars

            resp_chars = entry.get("response_chars")
            if not isinstance(resp_chars, int):
                resp_chars = len(_safe_json_dumps(entry.get("response")))
            response_char_sum += resp_chars

        summary_items: list[tuple[str, str]] = []
        summary_items.append(("Artifacts Dir", str(artifacts_dir)))
        summary_items.append(("Results File", str(results_path) if results else "-"))
        summary_items.append(("Report File", str(report_path) if report_entries else "-"))

        command = _get_nested(results, ["command"])
        if command:
            summary_items.append(("Command", str(command)))

        git_hash = _get_nested(results, ["git_hash"])
        if git_hash:
            summary_items.append(("Git Hash", str(git_hash)))

        model_id = _get_nested(results, ["target", "api_endpoint", "model_id"])
        if not model_id:
            model_id = _get_nested(run_config, ["target", "api_endpoint", "model_id"])
        if model_id:
            summary_items.append(("Model ID", str(model_id)))

        endpoint = _get_nested(results, ["target", "api_endpoint", "url"])
        if not endpoint:
            endpoint = _get_nested(run_config, ["target", "api_endpoint", "url"])
        if endpoint:
            summary_items.append(("Endpoint", str(endpoint)))

        task_name = _get_nested(run_config, ["task", "name"])
        if not task_name:
            task_name = _get_nested(run_config, ["task", "task_name"])
        if task_name:
            summary_items.append(("Task", str(task_name)))
        if model_set:
            summary_items.append(("Models", ", ".join(model_set)))

        metrics_rows = []
        results_metrics = _get_nested(results, ["results"])
        if isinstance(results_metrics, dict):
            for row in _flatten_metrics(results_metrics):
                metrics_rows.append(
                    [
                        row.get("path", "-"),
                        _format_value(row.get("value")),
                        _format_value(row.get("stats")),
                    ]
                )

        perf_rows = []
        if isinstance(eval_metrics, dict):
            for row in _flatten_numeric(eval_metrics):
                perf_rows.append([row.get("path", "-"), _format_value(row.get("value"))])

        artifact_rows: list[list[str]] = []
        for child in sorted(artifacts_dir.iterdir()):
            if child.is_file():
                size = child.stat().st_size
                artifact_rows.append([child.name, _format_bytes(size)])
            elif child.is_dir():
                try:
                    count = sum(1 for _ in child.iterdir())
                except Exception:
                    count = 0
                artifact_rows.append([f"{child.name}/", f"{count} items"])

        report_entries = report_entries[: self.sample_limit]

        stats_items: list[tuple[str, str]] = []
        stats_items.append(("Samples", str(sample_count)))
        if correct_count or incorrect_count:
            accuracy = correct_count / (correct_count + incorrect_count)
            stats_items.append(("Accuracy", f"{accuracy * 100:.1f}%"))
            stats_items.append(
                (
                    "Correct / Incorrect / Unknown",
                    f"{correct_count} / {incorrect_count} / {unknown_count}",
                )
            )
        stats_items.append(
            ("Errors", f"{error_count} ({(error_count / sample_count * 100):.1f}%)")
            if sample_count
            else ("Errors", "0")
        )
        stats_items.append(
            ("Tool Calls", f"{tool_count} ({(tool_count / sample_count * 100):.1f}%)")
            if sample_count
            else ("Tool Calls", "0")
        )
        if sample_count:
            stats_items.append(
                ("Avg Request Chars", f"{request_char_sum / sample_count:.0f}")
            )
            stats_items.append(
                ("Avg Response Chars", f"{response_char_sum / sample_count:.0f}")
            )
        if any(usage_totals.values()):
            stats_items.append(
                (
                    "Tokens (prompt / completion / total)",
                    f"{usage_totals['prompt_tokens']} / {usage_totals['completion_tokens']} / {usage_totals['total_tokens']}",
                )
            )
        if request_type_counts:
            request_type_summary = ", ".join(
                f"{key}:{value}" for key, value in sorted(request_type_counts.items())
            )
            stats_items.append(("Request Types", request_type_summary))

        finish_rows = [
            [reason, str(count)]
            for reason, count in sorted(
                finish_reason_counts.items(), key=lambda x: (-x[1], x[0])
            )
        ]

        payload = {
            "summary": summary_items,
            "stats": stats_items,
            "metrics_rows": metrics_rows,
            "perf_rows": perf_rows,
            "artifact_rows": artifact_rows,
            "samples": report_entries,
            "finish_rows": finish_rows,
            "raw_results": raw_results,
            "raw_run_config": raw_run_config,
            "raw_eval_metrics": raw_eval_metrics,
        }

        html_content = render_analysis_html(payload)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        print(f"Analysis report written to: {output_path}")
