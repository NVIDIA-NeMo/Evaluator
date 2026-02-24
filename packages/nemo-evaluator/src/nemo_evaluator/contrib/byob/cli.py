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

"""CLI for BYOB operations."""

import argparse
import glob
import os
import sys

import yaml

from nemo_evaluator.contrib.byob.compiler import compile_benchmark, install_benchmark


def _get_version():
    try:
        from nemo_evaluator.package_info import __version__
        return __version__
    except ImportError:
        return "unknown"


def byob_compile(args=None):
    """Compile a BYOB benchmark into a NeMo Evaluator plugin."""
    parser = argparse.ArgumentParser(
        description="Compile a BYOB benchmark into a NeMo Evaluator plugin"
    )
    parser.add_argument("module", nargs="?", help="Path to Python file with @benchmark decorators")
    parser.add_argument("--install-dir", default=None, help="Custom installation directory")
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {_get_version()}",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        default=False,
        help="List installed BYOB benchmark packages",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate benchmark and show configuration without installing",
    )
    parser.add_argument(
        "--check-requirements",
        action="store_true",
        default=False,
        help="After compilation, verify all declared requirements are importable",
    )
    parsed = parser.parse_args(args)

    # --list: show installed benchmarks
    if parsed.list:
        default_dir = os.path.expanduser("~/.nemo-evaluator/byob_packages/")
        search_dir = parsed.install_dir or default_dir

        if not os.path.exists(search_dir):
            print(f"No BYOB packages found at: {search_dir}")
            sys.exit(0)

        packages = []
        for fw_yml in glob.glob(os.path.join(search_dir, "*/core_evals/*/framework.yml")):
            try:
                with open(fw_yml) as f:
                    fw = yaml.safe_load(f)
                if fw:
                    name = fw.get("framework", {}).get("name", "unknown")
                    evals = fw.get("evaluations", [])
                    for ev in evals:
                        eval_type = ev.get("defaults", {}).get("config", {}).get("type", "unknown")
                        packages.append((name, eval_type))
            except Exception:
                continue

        if not packages:
            print(f"No BYOB packages found at: {search_dir}")
        else:
            print(f"Installed BYOB benchmarks ({search_dir}):")
            for name, eval_type in packages:
                print(f"  {eval_type}")
        sys.exit(0)

    # module is required for compile and dry-run
    if not parsed.module:
        parser.error("the following arguments are required: module")

    # --dry-run: validate without installing
    if parsed.dry_run:
        try:
            compiled = compile_benchmark(parsed.module)
        except Exception as e:
            print(f"VALIDATION FAILED: {e}", file=sys.stderr)
            sys.exit(1)

        print("Validation passed. Benchmarks found:")
        for name, fdf in compiled.items():
            eval_entry = fdf["evaluations"][0]
            print(f"  - {eval_entry['name']} (normalized: {name})")
            ds = fdf["defaults"]["config"]["params"]["extra"]["dataset"]
            print(f"    Dataset: {ds}")
            if os.path.exists(ds):
                with open(ds) as f:
                    sample_count = sum(1 for line in f if line.strip())
                print(f"    Samples: {sample_count}")
            else:
                print(f"    WARNING: Dataset not found: {ds}", file=sys.stderr)
            # Show requirements if declared
            reqs = fdf["defaults"]["config"]["params"]["extra"].get("requirements", [])
            if reqs:
                print(f"    Requirements: {', '.join(reqs)}")
        sys.exit(0)

    # Normal compile + install
    print(f"Compiling benchmarks from: {parsed.module}")
    compiled = compile_benchmark(parsed.module)

    for name, fdf in compiled.items():
        pkg_name = f"byob_{name}"
        print(f"\n  Benchmark: {fdf['evaluations'][0]['name']}")
        print(f"  Package:   {pkg_name}")

        pkg_dir = install_benchmark(name, fdf, install_dir=parsed.install_dir)
        eval_type = f"{pkg_name}.{fdf['evaluations'][0]['name']}"

        print(f"  Location:  {pkg_dir}")
        print(f"")
        print(f"  To run this benchmark:")
        print(f"    export PYTHONPATH=\"{os.path.dirname(pkg_dir)}:$PYTHONPATH\"")
        print(f"    nemo-evaluator run_eval \\")
        print(f"      --eval_type {eval_type} \\")
        print(f"      --model_url <YOUR_MODEL_URL> \\")
        print(f"      --model_id <YOUR_MODEL_ID>")

    # Check requirements if requested
    if parsed.check_requirements:
        from nemo_evaluator.contrib.byob.runner import check_requirements

        print("\nRequirements check:")
        any_failed = False
        for name, fdf in compiled.items():
            reqs = fdf["defaults"]["config"]["params"]["extra"].get("requirements", [])
            if not reqs:
                print(f"  {name}: no requirements declared")
                continue
            warnings = check_requirements(reqs)
            if warnings:
                any_failed = True
                for w in warnings:
                    print(f"  FAIL: {w}", file=sys.stderr)
            else:
                print(f"  {name}: all {len(reqs)} requirement(s) satisfied")
        if any_failed:
            print("\nSome requirements are not satisfied.", file=sys.stderr)
            sys.exit(1)

    print(f"\nCompiled {len(compiled)} benchmark(s) successfully.")


if __name__ == "__main__":
    byob_compile()
