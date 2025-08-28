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

from typing import ClassVar, Type

from nemo_evaluator.api.base import EvalFrameworkBase


class EvalFramworkRegistry:
    """Singleton registry for storing framework entry classes."""

    _instance: ClassVar["EvalFramworkRegistry | None"] = None
    _entries: list[Type[EvalFrameworkBase]]

    def __new__(cls) -> "EvalFramworkRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._entries = []
        return cls._instance

    def register(
        self, framework_entry_class: Type[EvalFrameworkBase]
    ) -> Type[EvalFrameworkBase]:
        if not issubclass(framework_entry_class, EvalFrameworkBase):
            raise TypeError(
                f"Class {framework_entry_class.__name__} must inherit from EntryBase"
            )

        # Fail fast if a framework definition is missing
        assert framework_entry_class.framework_def().exists()

        self._entries.append(framework_entry_class)
        return framework_entry_class

    def get_entries(self) -> list[Type[EvalFrameworkBase]]:
        # This is to prevent modification, those are lightweight
        return self._entries.copy()


# Global singleton instance
_REGISTRY = EvalFramworkRegistry()


# exposed API of this registry
def get_global_registry() -> EvalFramworkRegistry:
    return _REGISTRY


def register_framework(
    user_framework: Type[EvalFrameworkBase],
) -> Type[EvalFrameworkBase]:
    """Decorator to register an entry class for evaluation.

    Usage:
        @register_framework
        class ExampleEvalFramwork(EvalFrameworkBase):
            @staticmethod
            def parse_output(output_dir: str) -> EvaluationResult:
                # COMPLETE: add the parsing logic
                pass

            @staticmethod
            def framework_def() -> Path:
                # COMPLETE: add the path to your `framework.yml`
                pass
    """
    return get_global_registry().register(user_framework)
