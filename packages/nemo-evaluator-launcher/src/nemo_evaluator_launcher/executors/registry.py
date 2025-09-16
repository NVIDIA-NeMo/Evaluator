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
from typing import Callable, Type

from nemo_evaluator_launcher.executors.base import BaseExecutor

_EXECUTOR_REGISTRY: dict[str, Type[BaseExecutor]] = {}


def register_executor(
    executor_name: str,
) -> Callable[[Type[BaseExecutor]], Type[BaseExecutor]]:
    def wrapper(executor_cls: Type[BaseExecutor]) -> Type[BaseExecutor]:
        _EXECUTOR_REGISTRY[executor_name] = executor_cls
        return executor_cls

    return wrapper


def get_executor(executor_name: str) -> Type[BaseExecutor]:
    if executor_name not in _EXECUTOR_REGISTRY:
        raise ValueError(
            f"Executor {executor_name} not found. Available executors: {list(_EXECUTOR_REGISTRY.keys())}"
        )
    return _EXECUTOR_REGISTRY[executor_name]
