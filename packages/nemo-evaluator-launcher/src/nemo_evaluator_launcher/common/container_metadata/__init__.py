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
"""Container metadata management: registries, intermediate representations, and loading."""

from nemo_evaluator_launcher.common.container_metadata.intermediate_repr import (
    HarnessIntermediateRepresentation,
    TaskIntermediateRepresentation,
    load_harnesses_and_tasks_from_tasks_file,
    load_tasks_from_tasks_file,
)
from nemo_evaluator_launcher.common.container_metadata.registries import (
    DockerRegistryHandler,
    create_authenticator,
)
from nemo_evaluator_launcher.common.container_metadata.utils import (
    parse_container_image,
)

__all__ = [
    "DockerRegistryHandler",
    "create_authenticator",
    "HarnessIntermediateRepresentation",
    "TaskIntermediateRepresentation",
    "load_harnesses_and_tasks_from_tasks_file",
    "load_tasks_from_tasks_file",
    "parse_container_image",
]

# Optional imports:
# `loading` pulls in `nemo_evaluator` (and deps like `pydantic`). Keep IR-only
# workflows (e.g., docs autogen) usable without requiring the full stack.
try:
    from nemo_evaluator_launcher.common.container_metadata.loading import (  # noqa: F401
        extract_framework_yml,
        load_tasks_from_container,
        parse_framework_to_irs,
    )

    __all__.extend(
        [
            "extract_framework_yml",
            "load_tasks_from_container",
            "parse_framework_to_irs",
        ]
    )
except ModuleNotFoundError:
    # Allow importing this package for IR-only workflows (docs autogen, etc.)
    pass
