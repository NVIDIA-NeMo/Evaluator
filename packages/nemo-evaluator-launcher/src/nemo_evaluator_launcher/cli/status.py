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

import json
from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator_launcher.api.functional import get_status


@dataclass
class Cmd:
    """Status command configuration."""

    job_ids: list[str] = field(
        default_factory=list,
        positional=True,
    )
    json: bool = field(
        default=False,
        action="store_true",
        help="Print output as JSON instead of table format",
    )

    def execute(self):
        res = get_status(self.job_ids)
        if self.json:
            print(json.dumps(res, indent=2))
        else:
            self._print_table(res)

    def _print_table(self, jobs):
        """Print job status as a table."""
        if not jobs:
            print("No jobs found.")
            return

        # Define executor-specific mappings
        executor_mappings = {
            "slurm_job_id": ("Slurm Job ID", "remote_rundir_path"),
            "lepton_job_name": ("Lepton Job Name", "endpoint_name"),
            "pipeline_id": ("Pipeline ID", None),
            "container": ("Container", "output_dir"),
        }

        # Determine executor type and headers
        first_data = jobs[0].get("data", {}) if jobs else {}
        executor_key = next((k for k in executor_mappings if k in first_data), None)

        if executor_key:
            info_header, _ = executor_mappings[executor_key]
        else:
            info_header = "Executor Info"

        headers = ["Job ID", "Status", "Progress", info_header, "Location"]

        # Build rows
        rows = []
        for job in jobs:
            data = job.get("data", {})

            # Extract executor info
            executor_info = str(data.get(executor_key, "")) if executor_key else ""

            # Extract location
            if executor_key == "slurm_job_id":
                path = data.get("remote_rundir_path", "")
                location = (
                    "<output_dir>/" + "/".join(path.split("/")[-2:]) if path else ""
                )
            elif executor_key == "lepton_job_name":
                location = data.get("endpoint_name") or "shared-endpoint"
            elif executor_key == "pipeline_id":
                location = "gitlab-ci"
            elif executor_key == "container":
                path = data.get("output_dir", "")
                location = path.split("/")[-1] if path else ""
            else:
                location = ""

            rows.append(
                [
                    job.get("job_id", ""),
                    job.get("status", ""),
                    job.get("progress", ""),
                    executor_info,
                    location,
                ]
            )

        # Calculate column widths and print
        widths = [
            max(len(str(headers[i])), max(len(str(row[i])) for row in rows))
            for i in range(len(headers))
        ]

        header_row = " | ".join(
            headers[i].ljust(widths[i]) for i in range(len(headers))
        )
        print(header_row)
        print("-" * len(header_row))

        for row in rows:
            print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(row))))
