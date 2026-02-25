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

from nemo_evaluator.byob.compiler import compile_benchmark, install_benchmark


def byob_compile(args=None):
    """Compile a BYOB benchmark into a NeMo Evaluator plugin."""
    parser = argparse.ArgumentParser(
        description="Compile a BYOB benchmark into a NeMo Evaluator plugin"
    )
    parser.add_argument("module", help="Path to Python file with @benchmark decorators")
    parser.add_argument("--install-dir", default=None, help="Custom installation directory")
    parser.add_argument(
        "--native",
        action="store_true",
        default=False,
        help="Generate native-mode package (in-process execution, no subprocess)",
    )
    parsed = parser.parse_args(args)

    print(f"Compiling benchmarks from: {parsed.module}")
    execution_mode = "native" if parsed.native else "subprocess"
    compiled = compile_benchmark(parsed.module, execution_mode=execution_mode)

    for name, fdf in compiled.items():
        pkg_name = f"byob_{name}"
        print(f"  Installing {pkg_name}...")
        pkg_dir = install_benchmark(name, fdf, install_dir=parsed.install_dir)
        eval_type = f"{pkg_name}.{fdf['evaluations'][0]['name']}"
        print(f"  Installed to: {pkg_dir}")
        print(f"  Run with: nemo-evaluator run_eval --eval_type {eval_type}")

    print(f"\nCompiled {len(compiled)} benchmark(s).")


if __name__ == "__main__":
    byob_compile()
