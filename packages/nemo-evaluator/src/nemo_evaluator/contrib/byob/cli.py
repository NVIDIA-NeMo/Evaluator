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
from nemo_evaluator.contrib.byob.defaults import DEFAULT_BASE_IMAGE


def _pip_install_editable(pkg_dir: str) -> None:
    """Install a compiled BYOB package.

    Uses ``pip install`` (or ``uv pip install`` if available) so the
    package is immediately importable without PYTHONPATH manipulation.

    Note: non-editable install is required for correct namespace package
    merging when multiple BYOB packages share the ``nemo_evaluator`` namespace.
    Editable installs (``-e``) create per-package finders that shadow each
    other, breaking multi-package discovery.
    """
    import shutil
    import subprocess

    uv_bin = shutil.which("uv")
    if uv_bin:
        cmd = [uv_bin, "pip", "install", pkg_dir]
    else:
        cmd = [sys.executable, "-m", "pip", "install", pkg_dir]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  WARNING: Auto-install failed: {e.stderr.strip()}", file=sys.stderr)
        print(f'  Fallback: export PYTHONPATH="{pkg_dir}:$PYTHONPATH"', file=sys.stderr)


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
    parser.add_argument(
        "module", nargs="?", help="Path to Python file with @benchmark decorators"
    )
    parser.add_argument(
        "--install-dir", default=None, help="Custom installation directory"
    )
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
    parser.add_argument(
        "--no-install",
        action="store_true",
        default=False,
        help="Skip automatic pip install after compilation (requires manual PYTHONPATH setup)",
    )
    parser.add_argument(
        "--containerize",
        action="store_true",
        default=False,
        help="Build a Docker image from the compiled benchmark",
    )
    parser.add_argument(
        "--push",
        type=str,
        default=None,
        metavar="REGISTRY/IMAGE:TAG",
        help="Push the built image to a registry (implies --containerize)",
    )
    parser.add_argument(
        "--base-image",
        type=str,
        default=DEFAULT_BASE_IMAGE,
        help=f"Base Docker image for containerization (default: {DEFAULT_BASE_IMAGE})",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Docker image tag (default: byob_<name>:latest)",
    )
    parsed = parser.parse_args(args)

    # --push implies --containerize
    if parsed.push:
        parsed.containerize = True

    # --list: show installed benchmarks
    if parsed.list:
        default_dir = os.path.expanduser("~/.nemo-evaluator/byob_packages/")
        search_dir = parsed.install_dir or default_dir

        if not os.path.exists(search_dir):
            print(f"No BYOB packages found at: {search_dir}")
            sys.exit(0)

        packages = []
        for fw_yml in glob.glob(
            os.path.join(search_dir, "*/nemo_evaluator/*/framework.yml")
        ):
            try:
                with open(fw_yml) as f:
                    fw = yaml.safe_load(f)
                if fw:
                    name = fw.get("framework", {}).get("name", "unknown")
                    evals = fw.get("evaluations", [])
                    for ev in evals:
                        eval_type = (
                            ev.get("defaults", {})
                            .get("config", {})
                            .get("type", "unknown")
                        )
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

        # Auto-install the package so it's immediately discoverable
        if not parsed.no_install:
            _pip_install_editable(pkg_dir)
            print(f"  Installed: {pkg_name} (discoverable by nemo-evaluator)")
        else:
            print("")
            print("  NOTE: --no-install specified. To make discoverable, run:")
            print(f'    export PYTHONPATH="{pkg_dir}:$PYTHONPATH"')

        print("")
        print("  To run this benchmark:")
        print("    nemo-evaluator run_eval \\")
        print(f"      --eval_type {eval_type} \\")
        print("      --model_url <YOUR_MODEL_URL> \\")
        print("      --model_id <YOUR_MODEL_ID> \\")
        print("      --model_type chat \\")
        print("      --output_dir ./results \\")
        print("      --api_key_name <YOUR_API_KEY_ENV_VAR>")

    # Containerization if requested
    if parsed.containerize:
        import tempfile

        from nemo_evaluator.contrib.byob.containerize import (
            build_image,
            prepare_build_context,
            push_image,
        )

        for name, fdf in compiled.items():
            pkg_name = f"byob_{name}"
            pkg_dir = os.path.join(
                parsed.install_dir
                or os.path.expanduser("~/.nemo-evaluator/byob_packages/"),
                pkg_name,
            )
            tag = parsed.tag or parsed.push or f"{pkg_name}:latest"
            user_reqs = (
                fdf.get("defaults", {})
                .get("config", {})
                .get("params", {})
                .get("extra", {})
                .get("requirements", [])
            )

            with tempfile.TemporaryDirectory() as context_dir:
                prepare_build_context(pkg_dir, fdf, context_dir)
                build_image(
                    context_dir=context_dir,
                    tag=tag,
                    base_image=parsed.base_image,
                    pkg_name=pkg_name,
                    user_requirements=user_reqs or None,
                )
                print(f"  Docker image built: {tag}")

                if parsed.push:
                    push_tag = parsed.push
                    if push_tag != tag:
                        import subprocess

                        subprocess.run(["docker", "tag", tag, push_tag], check=True)
                    push_image(push_tag)
                    print(f"  Docker image pushed: {push_tag}")

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
