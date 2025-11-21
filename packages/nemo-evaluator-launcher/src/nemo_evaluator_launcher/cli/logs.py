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
"""Logs command for streaming logs from evaluation jobs."""

import sys
from dataclasses import dataclass
from typing import Callable, Dict

from simple_parsing import field

import nemo_evaluator_launcher.common.printing_utils as pu
from nemo_evaluator_launcher.api.functional import stream_logs
from nemo_evaluator_launcher.common.execdb import ExecutionDB
from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class Cmd:
    """Logs command configuration."""

    id: str = field(
        default="",
        positional=True,
        help="Invocation ID or job ID (e.g., '15b9f667' or '15b9f667.0')",
    )

    def execute(self) -> None:
        """Execute the logs command to stream logs from jobs."""
        if not self.id:
            logger.error("ID is required")
            sys.exit(1)

        db = ExecutionDB()

        # Determine if this is an invocation ID or job ID and validate it exists
        if "." in self.id:
            # This is a job ID - get single job
            job_data = db.get_job(self.id)
            if job_data is None:
                logger.error(f"Job {self.id} not found")
                sys.exit(1)
        else:
            # This is an invocation ID - get all jobs
            jobs = db.get_jobs(self.id)
            if not jobs:
                logger.error(f"Invocation {self.id} not found")
                sys.exit(1)

        # Build color mapping for job IDs
        colors = [pu.red, pu.green, pu.yellow, pu.magenta, pu.cyan]
        job_colors: Dict[str, Callable[[str], str]] = {}
        color_index = 0

        # Collect all job IDs that will be streamed
        if "." in self.id:
            job_ids = [self.id]
        else:
            job_ids = list(jobs.keys())

        for job_id in job_ids:
            job_colors[job_id] = colors[color_index % len(colors)]
            color_index += 1

        # Stream logs from executor
        try:
            log_stream = stream_logs(self.id)
            for job_id, task_name, log_line in log_stream:
                prefix = f"{task_name}:"
                color_func = job_colors.get(job_id, pu.grey)
                if log_line:
                    print(f"{color_func(prefix)} {log_line}")
                else:
                    # Print empty lines without prefix
                    print()

        except ValueError:
            # Handle case where executor doesn't support streaming
            # Warning already logged by BaseExecutor.stream_logs
            pass
        except KeyboardInterrupt:
            # Clean exit on Ctrl+C
            pass
        except Exception as e:
            logger.error(f"Error streaming logs: {e}")
            sys.exit(1)
