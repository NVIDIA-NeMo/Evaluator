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
"""Base executor interface and execution status types for nemo-evaluator-launcher.

Defines the abstract interface for all executor implementations and common status types.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from omegaconf import DictConfig


class ExecutionState(Enum):
    """Enumeration of possible execution states."""

    PENDING = "pending"
    RUNNING = "running"
    FAILED = "failed"
    SUCCESS = "success"
    KILLED = "killed"


@dataclass
class ExecutionStatus:
    """Represents the status of an execution."""

    id: str
    state: ExecutionState
    progress: Optional[dict[str, Any]] = None


class BaseExecutor(ABC):
    @staticmethod
    @abstractmethod
    def execute_eval(cfg: DictConfig, dry_run: bool = False) -> str:
        """Run an evaluation job using the provided configuration.

        Args:
            cfg: The configuration object for the evaluation run.
            dry_run: If True, prepare scripts and save them without execution.

        Returns:
            str: The invocation ID for the evaluation run.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def get_status(id: str) -> list[ExecutionStatus]:
        """Get the status of a job or invocation group by ID.

        Args:
            id: Unique job or invocation identifier.

        Returns:
            list[ExecutionStatus]: List of execution statuses for the job(s).

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def kill_job(job_id: str) -> None:
        """Kill a job by its ID.

        Args:
            job_id: The job ID to kill.

        Raises:
            ValueError: If job is not found or invalid.
            RuntimeError: If job cannot be killed.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError("Subclasses must implement this method")
