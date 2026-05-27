#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fix doubled path segments introduced during Sphinx-to-Fern MDX migration."""

from __future__ import annotations

import argparse
import pathlib
import sys

# Longest matches first so more-specific paths are fixed before prefixes.
REPLACEMENTS: tuple[tuple[str, str], ...] = (
    ("/about/about-concepts/", "/about/concepts/"),
    ("/about/about-concepts", "/about/concepts"),
    ("/about/about/key-features", "/about/key-features"),
    ("/about/about", "/about"),
    ("/evaluation/evaluation-benchmarks/", "/evaluation/benchmarks/"),
    ("/evaluation/evaluation-benchmarks", "/evaluation/benchmarks"),
    ("/evaluation/evaluation-run-evals/", "/evaluation/run-evals/"),
    ("/evaluation/evaluation-run-evals", "/evaluation/run-evals"),
    ("/evaluation/evaluation/", "/evaluation/"),
    ("/evaluation/evaluation", "/evaluation"),
    ("/libraries/libraries-nemo-evaluator-launcher/", "/libraries/nemo-evaluator-launcher/"),
    ("/libraries/libraries-nemo-evaluator-launcher", "/libraries/nemo-evaluator-launcher"),
    ("/libraries/libraries-nemo-evaluator/", "/libraries/nemo-evaluator/"),
    ("/libraries/libraries-nemo-evaluator", "/libraries/nemo-evaluator"),
    ("/get-started/get-started-quickstart/", "/get-started/quickstart/"),
    ("/get-started/get-started-quickstart", "/get-started/quickstart"),
    ("/get-started/get-started/install", "/get-started/install"),
    ("/get-started/get-started/", "/get-started/"),
    ("/get-started/get-started", "/get-started"),
    ("/tutorials/tutorials-how-to/", "/tutorials/how-to/"),
    ("/tutorials/tutorials-how-to", "/tutorials/how-to"),
    ("/tutorials/tutorials-nemo-fw/", "/tutorials/nemo-fw/"),
    ("/tutorials/tutorials-nemo-fw", "/tutorials/nemo-fw"),
    ("/model-deployment/deployment/", "/deployment/"),
    ("/model-deployment/", "/deployment/"),
    ("/references/references-api-nemo-evaluator/", "/references/api/nemo-evaluator/"),
    ("/references/references-api-nemo-evaluator", "/references/api/nemo-evaluator"),
    ('href="/api/nemo-evaluator-launcher/', 'href="/references/api/nemo-evaluator-launcher/'),
    ('](/api/nemo-evaluator-launcher/', '](/references/api/nemo-evaluator-launcher/'),
    ('href="/api/nemo-evaluator/', 'href="/references/api/nemo-evaluator/'),
    ('](/api/nemo-evaluator/', '](/references/api/nemo-evaluator/'),
)


def fix_content(content: str) -> str:
    for old, new in REPLACEMENTS:
        content = content.replace(old, new)
    return content


def fix_file(path: pathlib.Path, *, dry_run: bool) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = fix_content(original)
    if updated == original:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        type=pathlib.Path,
        help="Directories to scan (default: docs/ excluding fern/versions snapshots)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = pathlib.Path(__file__).resolve().parent
    docs_dir = script_dir.parent.parent
    roots = args.roots or [docs_dir]

    changed_files = 0
    for root in roots:
        root = root.resolve()
        for path in sorted(root.rglob("*.mdx")):
            if "fern/versions/" in path.as_posix():
                continue
            if fix_file(path, dry_run=args.dry_run):
                changed_files += 1
                print(path)

    action = "Would update" if args.dry_run else "Updated"
    print(f"{action} {changed_files} file(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
