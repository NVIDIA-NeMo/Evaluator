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
"""Run a lightweight log viewer for NeMo Evaluator results."""

from __future__ import annotations

from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from simple_parsing import field


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>NeMo Evaluator Viewer</title>
  <style>
    :root {
      --bg: #0f1116;
      --panel: #141824;
      --panel-2: #10131b;
      --text: #e8ecf1;
      --muted: #9aa6b2;
      --accent: #6f9bff;
      --border: rgba(255, 255, 255, 0.08);
      --shadow: 0 12px 28px rgba(0, 0, 0, 0.45);
    }
    body {
      margin: 0;
      font-family: "Space Grotesk", "IBM Plex Sans", "Source Sans 3", sans-serif;
      background: var(--bg);
      color: var(--text);
      height: 100vh;
      display: grid;
      grid-template-columns: 280px 1fr;
    }
    .sidebar {
      background: var(--panel);
      border-right: 1px solid var(--border);
      padding: 16px;
      overflow-y: auto;
    }
    .main {
      display: flex;
      flex-direction: column;
      height: 100vh;
    }
    h1 {
      font-size: 1.2rem;
      margin: 0 0 8px;
    }
    .list {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 16px;
    }
    .item {
      padding: 8px 10px;
      border-radius: 10px;
      background: var(--panel-2);
      border: 1px solid var(--border);
      cursor: pointer;
      font-size: 0.85rem;
      color: var(--text);
    }
    .item.active {
      background: var(--accent);
      color: #0b0f14;
      border-color: rgba(111, 155, 255, 0.6);
    }
    .toolbar {
      display: flex;
      gap: 8px;
      padding: 10px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
      align-items: center;
    }
    .select {
      background: var(--panel-2);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 8px;
      font-size: 0.85rem;
    }
    .tabs {
      display: flex;
      gap: 8px;
      padding: 8px 14px;
      background: var(--panel);
      border-bottom: 1px solid var(--border);
    }
    .tab {
      padding: 6px 12px;
      border-radius: 999px;
      border: 1px solid var(--border);
      background: var(--panel-2);
      color: var(--text);
      cursor: pointer;
      font-size: 0.85rem;
    }
    .tab.active {
      background: var(--accent);
      color: #0b0f14;
    }
    .content {
      flex: 1;
      overflow: hidden;
    }
    .pane {
      display: none;
      height: 100%;
    }
    .pane.active {
      display: block;
    }
    iframe {
      border: 0;
      width: 100%;
      height: 100%;
    }
    pre {
      margin: 0;
      padding: 16px;
      color: var(--text);
      font-family: "IBM Plex Mono", ui-monospace, monospace;
      font-size: 0.82rem;
      white-space: pre-wrap;
    }
    .empty {
      padding: 24px;
      color: var(--muted);
    }
  </style>
</head>
<body>
  <aside class="sidebar">
    <h1>Runs</h1>
    <div id="runList" class="list"></div>
    <h1>Tasks</h1>
    <div id="taskList" class="list"></div>
  </aside>
  <main class="main">
    <div class="toolbar">
      <select id="runSelect" class="select"></select>
      <select id="taskSelect" class="select"></select>
      <span id="status" style="color: var(--muted); font-size: 0.85rem;"></span>
    </div>
    <div class="tabs">
      <button class="tab active" data-pane="reportPane">Report</button>
      <button class="tab" data-pane="logsPane">Logs</button>
      <button class="tab" data-pane="rawPane">Raw</button>
    </div>
    <div class="content">
      <div id="reportPane" class="pane active">
        <iframe id="reportFrame" src=""></iframe>
      </div>
      <div id="logsPane" class="pane">
        <div class="toolbar">
          <select id="logSelect" class="select"></select>
        </div>
        <pre id="logOutput" class="empty">Select a log file.</pre>
      </div>
      <div id="rawPane" class="pane">
        <pre id="rawOutput" class="empty">Select a task to view raw files.</pre>
      </div>
    </div>
  </main>
  <script>
    const runList = document.getElementById('runList');
    const taskList = document.getElementById('taskList');
    const runSelect = document.getElementById('runSelect');
    const taskSelect = document.getElementById('taskSelect');
    const status = document.getElementById('status');
    const reportFrame = document.getElementById('reportFrame');
    const logSelect = document.getElementById('logSelect');
    const logOutput = document.getElementById('logOutput');
    const rawOutput = document.getElementById('rawOutput');
    const tabs = document.querySelectorAll('.tab');
    const panes = document.querySelectorAll('.pane');

    async function fetchJson(url) {
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      return await res.json();
    }

    function setActive(list, value) {
      list.querySelectorAll('.item').forEach((el) => {
        if (el.dataset.value === value) el.classList.add('active');
        else el.classList.remove('active');
      });
    }

    async function loadRuns() {
      const data = await fetchJson('/api/runs');
      runList.innerHTML = '';
      runSelect.innerHTML = '';
      data.runs.forEach((run) => {
        const item = document.createElement('div');
        item.className = 'item';
        item.textContent = run;
        item.dataset.value = run;
        item.onclick = () => selectRun(run);
        runList.appendChild(item);
        const opt = document.createElement('option');
        opt.value = run;
        opt.textContent = run;
        runSelect.appendChild(opt);
      });
      if (data.runs.length) {
        selectRun(data.runs[0]);
      }
    }

    async function selectRun(run) {
      runSelect.value = run;
      setActive(runList, run);
      status.textContent = `Run: ${run}`;
      const data = await fetchJson(`/api/run/${encodeURIComponent(run)}/tasks`);
      taskList.innerHTML = '';
      taskSelect.innerHTML = '';
      data.tasks.forEach((task) => {
        const item = document.createElement('div');
        item.className = 'item';
        item.textContent = task;
        item.dataset.value = task;
        item.onclick = () => selectTask(run, task);
        taskList.appendChild(item);
        const opt = document.createElement('option');
        opt.value = task;
        opt.textContent = task;
        taskSelect.appendChild(opt);
      });
      if (data.tasks.length) {
        selectTask(run, data.tasks[0]);
      }
    }

    async function selectTask(run, task) {
      taskSelect.value = task;
      setActive(taskList, task);
      status.textContent = `Run: ${run} • Task: ${task}`;

      const report = await fetchJson(`/api/run/${encodeURIComponent(run)}/task/${encodeURIComponent(task)}/report`);
      reportFrame.src = report.url || '';

      const logs = await fetchJson(`/api/run/${encodeURIComponent(run)}/task/${encodeURIComponent(task)}/logs`);
      logSelect.innerHTML = '';
      logs.files.forEach((file) => {
        const opt = document.createElement('option');
        opt.value = file;
        opt.textContent = file;
        logSelect.appendChild(opt);
      });
      if (logs.files.length) {
        logSelect.value = logs.files[0];
        await loadLog();
      } else {
        logOutput.textContent = 'No logs found for this task.';
      }

      const raw = await fetchJson(`/api/run/${encodeURIComponent(run)}/task/${encodeURIComponent(task)}/raw`);
      rawOutput.textContent = raw.text || 'No raw files found.';
    }

    async function loadLog() {
      const run = runSelect.value;
      const task = taskSelect.value;
      const file = logSelect.value;
      if (!file) return;
      const data = await fetchJson(`/api/run/${encodeURIComponent(run)}/task/${encodeURIComponent(task)}/log?file=${encodeURIComponent(file)}`);
      logOutput.textContent = data.text || '';
    }

    runSelect.addEventListener('change', () => selectRun(runSelect.value));
    taskSelect.addEventListener('change', () => selectTask(runSelect.value, taskSelect.value));
    logSelect.addEventListener('change', () => loadLog());

    tabs.forEach((tab) => {
      tab.addEventListener('click', () => {
        tabs.forEach((btn) => btn.classList.remove('active'));
        tab.classList.add('active');
        const target = tab.dataset.pane;
        panes.forEach((pane) => {
          pane.classList.toggle('active', pane.id === target);
        });
      });
    });

    setInterval(() => {
      if (document.getElementById('logsPane').classList.contains('active')) {
        loadLog().catch(() => {});
      }
    }, 4000);

    loadRuns().catch((err) => {
      runList.innerHTML = `<div class=\"empty\">Failed to load runs: ${err}</div>`;
    });
  </script>
</body>
</html>
"""


def _safe_join(root: Path, *parts: str) -> Path:
    joined = root.joinpath(*parts).resolve()
    if not str(joined).startswith(str(root.resolve())):
        raise ValueError("Invalid path")
    return joined


def _find_runs(log_dir: Path, recursive: bool) -> list[str]:
    runs: list[str] = []
    if not log_dir.exists():
        return runs
    if not recursive:
        for child in sorted(log_dir.iterdir()):
            if child.is_dir():
                runs.append(child.name)
        return runs

    for child in sorted(log_dir.rglob("*")):
        if child.is_dir():
            # Detect a run by presence of at least one task dir with artifacts
            if any((task / "artifacts").is_dir() for task in child.iterdir() if task.is_dir()):
                rel = child.relative_to(log_dir).as_posix()
                runs.append(rel)
    return runs


def _find_tasks(run_dir: Path) -> list[str]:
    tasks: list[str] = []
    if not run_dir.exists():
        return tasks
    for child in sorted(run_dir.iterdir()):
        if child.is_dir() and (child / "artifacts").is_dir():
            tasks.append(child.name)
    return tasks


def _find_logs(task_dir: Path) -> list[str]:
    candidates: list[Path] = []
    logs_dir = task_dir / "logs"
    if logs_dir.is_dir():
        candidates.extend([p for p in logs_dir.iterdir() if p.is_file()])
    for child in task_dir.iterdir():
        if child.is_file() and child.suffix in {".log", ".txt", ".out"}:
            candidates.append(child)
    return [p.name for p in sorted(set(candidates))]


def _tail_file(path: Path, max_lines: int = 400) -> str:
    if not path.exists():
        return ""
    lines: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            lines.append(line.rstrip("\n"))
    return "\n".join(lines[-max_lines:])


@dataclass
class Cmd:
    """Start a NeMo Evaluator log viewer in the browser."""

    log_dir: str = field(
        default="nel-results",
        alias=["--log-dir"],
        help="Directory containing evaluator run outputs.",
    )
    host: str = field(
        default="127.0.0.1",
        alias=["--host"],
        help="Host interface to bind the viewer.",
    )
    port: int = field(
        default=7575,
        alias=["--port"],
        help="Port to bind the viewer.",
    )
    recursive: bool = field(
        default=True,
        alias=["--recursive"],
        help="Recursively scan log_dir for runs.",
    )

    def execute(self) -> None:
        log_root = Path(self.log_dir).expanduser().resolve()

        class Handler(BaseHTTPRequestHandler):
            def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_text(self, text: str, content_type: str = "text/plain") -> None:
                body = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:  # noqa: N802
                parsed = urlparse(self.path)
                if parsed.path == "/":
                    self._send_text(HTML_PAGE, "text/html")
                    return
                if parsed.path == "/api/runs":
                    runs = _find_runs(log_root, self.server.recursive)
                    self._send_json({"runs": runs})
                    return
                if parsed.path.startswith("/api/run/") and parsed.path.endswith("/tasks"):
                    parts = parsed.path.split("/")
                    run_rel = "/".join(parts[3:-1])
                    run_dir = _safe_join(log_root, run_rel)
                    tasks = _find_tasks(run_dir)
                    self._send_json({"tasks": tasks})
                    return
                if parsed.path.startswith("/api/run/") and parsed.path.endswith("/report"):
                    parts = parsed.path.split("/")
                    run_rel = "/".join(parts[3:-2])
                    task = parts[-2]
                    report = _safe_join(log_root, run_rel, task, "artifacts", "report.html")
                    url = f"/file?path={report.relative_to(log_root).as_posix()}" if report.exists() else ""
                    self._send_json({"url": url})
                    return
                if parsed.path.startswith("/api/run/") and parsed.path.endswith("/logs"):
                    parts = parsed.path.split("/")
                    run_rel = "/".join(parts[3:-2])
                    task = parts[-2]
                    task_dir = _safe_join(log_root, run_rel, task)
                    files = _find_logs(task_dir)
                    self._send_json({"files": files})
                    return
                if parsed.path.startswith("/api/run/") and parsed.path.endswith("/raw"):
                    parts = parsed.path.split("/")
                    run_rel = "/".join(parts[3:-2])
                    task = parts[-2]
                    artifacts = _safe_join(log_root, run_rel, task, "artifacts")
                    files = []
                    for name in ("results.yml", "run_config.yml", "eval_factory_metrics.json"):
                        path = artifacts / name
                        if path.exists():
                            files.append(f"{name}\\n{'-'*40}\\n{path.read_text(encoding='utf-8', errors='ignore')}")
                    self._send_json({"text": \"\\n\\n\".join(files)})
                    return
                if parsed.path.startswith("/api/run/") and parsed.path.endswith("/log"):
                    parts = parsed.path.split("/")
                    run_rel = "/".join(parts[3:-2])
                    task = parts[-2]
                    qs = parse_qs(parsed.query)
                    file_name = qs.get("file", [""])[0]
                    task_dir = _safe_join(log_root, run_rel, task)
                    logs_dir = task_dir / "logs"
                    path = logs_dir / file_name if logs_dir.is_dir() else task_dir / file_name
                    text = _tail_file(path)
                    self._send_json({"text": text})
                    return
                if parsed.path == "/file":
                    qs = parse_qs(parsed.query)
                    rel = qs.get("path", [""])[0]
                    try:
                        path = _safe_join(log_root, rel)
                    except Exception:
                        self.send_error(400)
                        return
                    if not path.exists():
                        self.send_error(404)
                        return
                    ext = path.suffix.lower()
                    content_type = "text/plain"
                    if ext == ".html":
                        content_type = "text/html"
                    elif ext == ".json":
                        content_type = "application/json"
                    elif ext in {".yml", ".yaml"}:
                        content_type = "text/yaml"
                    self._send_text(path.read_text(encoding="utf-8", errors="ignore"), content_type)
                    return
                self.send_error(404)

        server = ThreadingHTTPServer((self.host, self.port), Handler)
        server.recursive = self.recursive

        print(f"NeMo Evaluator Viewer running at http://{self.host}:{self.port}")
        print(f"Log directory: {log_root}")
        print("Press Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping viewer.")
            server.server_close()
