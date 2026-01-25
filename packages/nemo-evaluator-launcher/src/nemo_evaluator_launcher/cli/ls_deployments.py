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
"""List available deployments command."""

from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator_launcher.common.metadata import (
    DEPLOYMENT_CLI_FLAGS,
    DEPLOYMENT_DESCRIPTIONS,
    get_available_deployments,
)
from nemo_evaluator_launcher.common.printing_utils import bold, cyan, grey, magenta


# Deployment definitions with metadata (dynamically built from configs/deployment/)
DEPLOYMENTS = {
    name: {
        "description": DEPLOYMENT_DESCRIPTIONS.get(name, ""),
        "required": DEPLOYMENT_CLI_FLAGS.get(name, {}).get("required", []),
        "optional": DEPLOYMENT_CLI_FLAGS.get(name, {}).get("optional", []),
        "default": name == "none",
        **({"note": "Default URL: NVIDIA API Catalog"} if name == "none" else {}),
    }
    for name in get_available_deployments()
}


@dataclass
class Cmd:
    """List available deployments."""

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

        output = {"deployments": DEPLOYMENTS}
        print(json.dumps(output, indent=2))

    def _print_formatted(self) -> None:
        print()
        print(bold("Deployments (how model is served):"))
        print()

        for name, info in DEPLOYMENTS.items():
            default_marker = " (default)" if info["default"] else ""
            print(f"  {cyan(name)}{grey(default_marker)}")
            print(f"    {info['description']}")

            if info["required"]:
                required_str = ", ".join(info["required"])
                print(f"    {magenta('Required:')} {required_str}")

            if info["optional"]:
                optional_str = ", ".join(info["optional"])
                print(f"    {grey('Optional:')} {optional_str}")

            if info.get("note"):
                print(f"    {grey('Note:')} {info['note']}")

            print()

        print(grey("Use --deployment <name> to select a deployment."))
        print()
        print(grey("Examples:"))
        print(grey("  # Use NVIDIA API Catalog (default, no deployment needed)"))
        print(grey("  nel run --model meta/llama-3.2-3b-instruct --task ifeval"))
        print()
        print(grey("  # Deploy with vLLM"))
        print(grey("  nel run --deployment vllm --checkpoint /path/to/model --model-name my-model --task ifeval"))
        print()
