#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate Harbor registry override shards from upstream sources.

Run from the repo root:

    python harbor_datasets/registry_overrides/regenerate.py terminal_bench_2_1

Each shard is a JSON list of ``DatasetSpec`` dicts merged into the base
Harbor registry by ``nemo_evaluator.environments.harbor._load_registry``.
"""

from __future__ import annotations

import json
import sys
import tomllib
import urllib.request
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return resp.read()


def gen_terminal_bench_2_1() -> dict:
    git_url = "https://github.com/harbor-framework/terminal-bench-2-1.git"
    commit = "c5ee500c185224c97cd6caff7866a990a0057f41"
    manifest_url = f"https://raw.githubusercontent.com/harbor-framework/terminal-bench-2-1/{commit}/tasks/dataset.toml"
    manifest = tomllib.loads(_fetch(manifest_url).decode("utf-8"))
    task_names = [t["name"].rsplit("/", 1)[-1] for t in manifest["tasks"]]

    return {
        "name": "terminal-bench/terminal-bench-2-1",
        "version": "1.0",
        "description": (
            "Version 2.1 of Terminal-Bench, a benchmark for evaluating agents "
            "in terminal environments.  Vendored override pinned at "
            f"{commit} pending upstream registry entry."
        ),
        "tasks": [
            {
                "name": name,
                "git_url": git_url,
                "git_commit_id": commit,
                "path": f"tasks/{name}",
            }
            for name in task_names
        ],
    }


_GENERATORS = {
    "terminal_bench_2_1": gen_terminal_bench_2_1,
}


def main(args: list[str]) -> int:
    if not args:
        print("usage: regenerate.py <shard-name>", file=sys.stderr)
        print(f"available: {sorted(_GENERATORS)}", file=sys.stderr)
        return 2

    for name in args:
        gen = _GENERATORS.get(name)
        if gen is None:
            print(f"unknown shard: {name}", file=sys.stderr)
            return 2
        entry = gen()
        out_path = OUTPUT_DIR / f"{name}.json"
        out_path.write_text(json.dumps([entry], indent=2) + "\n")
        print(f"wrote {out_path} ({len(entry['tasks'])} tasks)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
