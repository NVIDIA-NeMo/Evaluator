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
"""Task loading utilities for loading tasks from containers."""

from typing import Optional

from nemo_evaluator_launcher.common.framework_loader import (
    extract_framework_yml,
    parse_framework_to_irs,
)
from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.common.task_ir import (
    TaskIntermediateRepresentation,
    _extract_harness_from_container,
)


def load_tasks_from_container(
    container: str,
    max_layer_size: Optional[int] = None,
    use_cache: bool = True,
) -> list[TaskIntermediateRepresentation]:
    """Load tasks from container by extracting and parsing framework.yml.

    Args:
        container: Container image identifier (e.g., nvcr.io/nvidia/eval-factory/simple-evals:25.10)
        max_layer_size: Optional maximum layer size in bytes for layer inspection
        use_cache: Whether to use caching for framework.yml extraction

    Returns:
        List of TaskIntermediateRepresentation objects

    Raises:
        ValueError: If container filtering results in no tasks
    """
    logger.debug(
        "Loading tasks from container",
        container=container,
    )

    # Extract harness name from container
    harness_name = _extract_harness_from_container(container)

    # Extract framework.yml from container (uses existing cache)
    framework_content, container_digest = extract_framework_yml(
        container=container,
        harness_name=harness_name,
        max_layer_size=max_layer_size or (100 * 1024),  # 100KB default
        use_cache=use_cache,
    )

    if not framework_content:
        logger.error(
            "Could not extract framework.yml from container",
            container=container,
        )
        return []

    # Parse framework.yml to IRs
    try:
        harness_ir, task_irs = parse_framework_to_irs(
            framework_content=framework_content,
            container_id=container,
            container_digest=container_digest,
            harness_name=harness_name,
        )
        logger.info(
            "Loaded tasks from container",
            container=container,
            num_tasks=len(task_irs),
        )

        # Verify all tasks are from the specified container (safeguard)
        expected_container = container
        original_count = len(task_irs)
        filtered_tasks = [
            task for task in task_irs if task.container == expected_container
        ]
        if len(filtered_tasks) != original_count:
            logger.warning(
                "Filtered out tasks from different containers",
                expected_container=expected_container,
                original_count=original_count,
                filtered_count=len(filtered_tasks),
            )
        if not filtered_tasks:
            error_msg = (
                f"No tasks found from specified container after filtering: {expected_container}"
            )
            logger.warning(
                "No tasks found from specified container after filtering",
                container=expected_container,
            )
            raise ValueError(error_msg)

        return filtered_tasks
    except Exception as e:
        logger.error(
            "Failed to parse framework.yml from container",
            container=container,
            error=str(e),
            exc_info=True,
        )
        return []
