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

from abc import ABC, abstractmethod
from pathlib import Path

from nemo_evaluator.api.api_dataclasses import EvaluationResult


class EvalFrameworkBase(ABC):
    """When subclassing, register with @register_framework decorator

    # register_framework is imported from nemo_evaluator.api.run
    """

    @staticmethod
    @abstractmethod
    def parse_output(output_dir: str) -> EvaluationResult:
        """Parser of the harness output into Nvidia Eval result type.

        Defining this method ensures that Nvidia Eval Commons can translate the harness's output
        into the dataclass understandable by the library.
        """
        pass

    @staticmethod
    @abstractmethod
    def framework_def() -> Path:
        """Path to the yml definition of the tasks.

        The file `framework.yml` contains the definition of tasks, default parameters, and other crucial properties
        of the harness. It's documented in the contributing guide and, for quick bootstrapping, can be created by
        `nemo_evaluator_example` command that is installed along with `nemo_evaluator`.
        """
        pass
