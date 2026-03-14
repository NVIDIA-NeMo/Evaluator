# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""
Reads environment variables and emits a CLI argument string for evaluation_with_nemo_run.py.

Usage (from within scripts/):
    python evaluation_with_nemo_run.py $(python argument_builder.py)

Each CLI argument --foo_bar maps to the environment variable FOO_BAR.
StoreTrueAction flags are emitted only when the env var is "true"/"1"/"yes"/"on".
"""

import argparse
import os

from evaluation_with_nemo_run import get_parser


def normalize_arg_name(arg_name: str) -> str:
    """Convert a CLI flag (e.g. '--tensor_parallelism_size') to its env var name ('TENSOR_PARALLELISM_SIZE')."""
    return arg_name.lstrip("-").upper().replace("-", "_")


def build_cli_args_from_env_vars(parser: argparse.ArgumentParser) -> str:
    """Inspect the parser, check env vars, and build a CLI argument string."""
    cli_parts = []

    for action in parser._actions:
        if not action.option_strings:
            continue

        long_flag = action.option_strings[-1]
        env_var = normalize_arg_name(long_flag)
        value = os.getenv(env_var)

        if not value:  # skip None and empty string
            continue

        if isinstance(action, argparse._StoreTrueAction):
            if value.lower() in ("true", "1", "yes", "on"):
                cli_parts.append(long_flag)
        else:
            cli_parts.extend([long_flag, value])

    return " ".join(cli_parts)


if __name__ == "__main__":
    print(build_cli_args_from_env_vars(get_parser()))
