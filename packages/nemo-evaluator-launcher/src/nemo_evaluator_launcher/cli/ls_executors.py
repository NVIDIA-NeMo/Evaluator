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
"""List available executors command."""

from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator_launcher.common.metadata import (
    EXECUTOR_CLI_FLAGS,
    EXECUTOR_DESCRIPTIONS,
    get_available_executors,
)
from nemo_evaluator_launcher.common.printing_utils import bold, cyan, grey, magenta


# Executor definitions with metadata (dynamically built from registry)
EXECUTORS = {
    name: {
        "description": EXECUTOR_DESCRIPTIONS.get(name, ""),
        "required": EXECUTOR_CLI_FLAGS.get(name, {}).get("required", []),
        "optional": EXECUTOR_CLI_FLAGS.get(name, {}).get("optional", []),
        "default": name == "local",
    }
    for name in get_available_executors()
}


@dataclass
class Cmd:
    """List available executors."""

    json: bool = field(
        default=False,
        action="store_true",
        help="Print output as JSON instead of formatted text",
    )

    def execute(self) -> None:
        if self.json:
            self._print_json()
        else:
            self._print_formatted()

    def _print_json(self) -> None:
        import json

        output = {"executors": EXECUTORS}
        print(json.dumps(output, indent=2))

    def _print_formatted(self) -> None:
        print()
        print(bold("Executors (where to run):"))
        print()

        for name, info in EXECUTORS.items():
            default_marker = " (default)" if info["default"] else ""
            print(f"  {cyan(name)}{grey(default_marker)}")
            print(f"    {info['description']}")

            if info["required"]:
                required_str = ", ".join(info["required"])
                print(f"    {magenta('Required:')} {required_str}")

            if info["optional"]:
                optional_str = ", ".join(info["optional"])
                print(f"    {grey('Optional:')} {optional_str}")

            print()

        print(grey("Use --executor <name> to select an executor."))
        print(grey("Example: nel run --executor slurm --slurm-hostname my-cluster ..."))
        print()
