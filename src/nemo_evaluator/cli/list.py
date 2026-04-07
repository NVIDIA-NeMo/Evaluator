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
"""nel list — unified benchmark discovery across all sources."""

from __future__ import annotations

import click


@click.command("list")
@click.option(
    "--source",
    "-s",
    type=click.Choice(["all", "builtin", "skills", "lm-eval"]),
    default="all",
    help="Filter benchmarks by source",
)
@click.option("--data-dir", default=None, help="NeMo Skills data directory (for --source skills)")
def list_cmd(source, data_dir):
    """List available benchmarks."""
    sections: list[tuple[str, list[str]]] = []

    if source in ("all", "builtin"):
        sections.append(("Built-in benchmarks", _list_builtin()))

    if source in ("all", "skills"):
        sections.append(("NeMo Skills benchmarks", _list_skills(data_dir)))

    if source in ("all", "lm-eval"):
        sections.append(("lm-eval-harness tasks", _list_lm_eval()))

    for title, items in sections:
        click.echo(f"\n{title}:")
        if not items:
            click.echo("  (none available)")
        for item in items:
            click.echo(f"  {item}")

    click.echo()


def _list_builtin() -> list[str]:
    from nemo_evaluator.environments.registry import list_environments

    names = list_environments()
    return [f"{name:30s} nel eval run --bench {name}" for name in names]


def _list_skills(data_dir: str | None) -> list[str]:
    try:
        from nemo_evaluator.environments.skills import list_skills_benchmarks
    except ImportError:
        return ["(nemo-skills not installed — pip install nemo-skills)"]

    benchmarks = list_skills_benchmarks(data_dir)
    if not benchmarks:
        return ["(none — run: ns prepare_data <benchmark>)"]

    items = []
    for b in benchmarks:
        splits = ", ".join(b["splits"])
        items.append(f"skills://{b['benchmark']:25s} type={b['metrics_type']:15s} splits=[{splits}]")
    return items


def _list_lm_eval() -> list[str]:
    try:
        from lm_eval.tasks import TaskManager

        tm = TaskManager()
        all_tasks = sorted(tm.all_tasks)
    except ImportError:
        return ["(lm-eval not installed — pip install lm-eval)"]
    except Exception:
        return ["(failed to load lm-eval tasks)"]

    generate_tasks = []
    for name in all_tasks:
        try:
            cfg = tm.get_task_dict([name])
            task_obj = list(cfg.values())[0] if cfg else None
            if task_obj is None:
                continue
            output_type = getattr(task_obj, "OUTPUT_TYPE", None)
            if output_type == "generate_until":
                generate_tasks.append(f"lm-eval://{name}")
        except Exception:
            continue

    if not generate_tasks:
        return [f"lm-eval://<task>   ({len(all_tasks)} tasks available, use: nel eval run --bench lm-eval://<task>)"]

    if len(generate_tasks) > 50:
        shown = generate_tasks[:50]
        shown.append(f"... and {len(generate_tasks) - 50} more")
        return shown
    return generate_tasks
