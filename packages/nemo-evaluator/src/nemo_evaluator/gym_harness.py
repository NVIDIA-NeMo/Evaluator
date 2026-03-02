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

"""
GymHarness interface for integrating evaluation harnesses with nemo-gym.

Each harness package (e.g., simple-evals, lm-eval) implements this interface
in a module named ``gym_harness.py`` at the package root, exporting a class
named ``Harness``.

The generic resource server in nemo-gym discovers implementations via::

    import {pkg_name}.gym_harness
    harness = {pkg_name}.gym_harness.Harness(eval_type=..., **kwargs)

``pkg_name`` is read from the harness's existing ``framework.yml``.
"""

from abc import ABC, abstractmethod
from typing import Any


class GymHarness(ABC):
    """Interface for making any eval harness available as a nemo-gym resource server.

    Subclasses must implement :meth:`get_dataset` (load benchmark data and
    format prompts) and :meth:`verify` (extract and score a model response).

    Implementations should be exported as ``Harness`` in
    ``{pkg_name}/gym_harness.py`` so the resource server can discover them
    by convention.
    """

    def __init__(self, eval_type: str, **kwargs: Any) -> None:
        self.eval_type = eval_type

    @abstractmethod
    def get_dataset(self) -> list[dict]:
        """Load benchmark data, format prompts, return JSONL-ready rows.

        Each row must contain at minimum:

        - ``responses_create_params``: ``{"input": [{"role": "user", "content": "..."}]}``
        - ``expected_answer``: ``str``

        Additional fields are passed through to :meth:`verify` as keyword
        arguments during evaluation.
        """
        ...

    @abstractmethod
    async def verify(
        self,
        response_text: str,
        expected_answer: str,
        **kwargs: Any,
    ) -> tuple[float, str | None]:
        """Extract an answer from *response_text* and score it against *expected_answer*.

        Returns ``(reward, extracted_answer)``.  ``extracted_answer`` may be
        ``None`` if extraction failed.

        This method is async to support judge-model fallback calls.
        """
        ...
