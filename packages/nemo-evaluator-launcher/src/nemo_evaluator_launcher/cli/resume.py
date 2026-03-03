# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""CLI command for resuming evaluations by re-executing existing scripts."""

from dataclasses import dataclass

from simple_parsing import field


@dataclass
class Cmd:
    """Resume command configuration."""

    invocation_id: str = field(
        positional=True,
        metadata={"help": "Invocation ID to resume (supports partial IDs)"},
    )

    def execute(self) -> None:
        """Execute the resume command."""
        from nemo_evaluator_launcher.api.functional import resume_eval
        from nemo_evaluator_launcher.common.printing_utils import bold, cyan, green

        try:
            resumed_invocation_id = resume_eval(self.invocation_id)
            print(
                bold(cyan("To check status:"))
                + f" nemo-evaluator-launcher status {resumed_invocation_id}"
            )
            print(
                bold(cyan("To view job info:"))
                + f" nemo-evaluator-launcher info {resumed_invocation_id}"
            )
            print(green(bold(f"✓ Resumed invocation {resumed_invocation_id}")))
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}")
            exit(1)
