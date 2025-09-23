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
from dataclasses import dataclass

from simple_parsing import field


@dataclass
class Cmd:
    """Kill command configuration."""

    id: str = field(
        positional=True,
        metadata={
            "help": "Job ID (e.g., aefc4819.0) or invocation ID (e.g., aefc4819) to kill"
        },
    )

    def execute(self) -> None:
        """Execute the kill command."""
        # Import heavy dependencies only when needed
        import json

        from nemo_evaluator_launcher.api.functional import kill_job_or_invocation

        result = kill_job_or_invocation(self.id)
        # Output as JSON
        print(json.dumps(result, indent=2))
