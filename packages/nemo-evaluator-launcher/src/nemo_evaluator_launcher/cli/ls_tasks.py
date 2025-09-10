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
import json
from dataclasses import dataclass

from nemo_evaluator_launcher.api.functional import get_tasks_list


@dataclass
class Cmd:
    """List command configuration."""

    def execute(self) -> None:
        # TODO(dfridman): modify `get_tasks_list` to return a list of dicts in the first place
        data = get_tasks_list()
        headers = ["task", "endpoint_type", "harness", "container"]
        supported_benchmarks = []
        for task_data in data:
            assert len(task_data) == len(headers)
            supported_benchmarks.append(dict(zip(headers, task_data)))
        print(json.dumps(supported_benchmarks, indent=2))
