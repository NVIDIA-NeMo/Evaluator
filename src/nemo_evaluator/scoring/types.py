# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Scoring data types shared across all scorer implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox


@dataclass
class ScorerInput:
    """Input passed to scorer functions and object-style metrics.

    Two access idioms are supported:

    **NEL-native (function-style scorers)** — named fields::

        sample.response           # model output
        sample.target             # ground truth
        sample.metadata[...]      # per-row dataset extras
        sample.config[...]        # per-benchmark settings (legacy)

    **NMP-native (object-style metrics)** — dict namespaces::

        sample.item[...]          # full dataset row (target + metadata)
        sample.sample[...]        # full inference payload (output_text, response)

    The dict namespaces (:attr:`item`, :attr:`sample`) are derived from the
    named fields; there is a single source of truth. The :meth:`jinja_context`
    helper produces a rendering context compatible with both NEL-native
    (``{{ response }}``, ``{{ target }}``, ``{{ metadata.x }}``) and
    NMP-native (``{{ item.reference }}``, ``{{ sample.output_text }}``)
    template vocabularies — so Jinja templates authored against either shape
    work without modification.

    The ``sandbox`` field is NEL-specific (no NMP equivalent) — for scorers
    that need to execute commands inside a per-problem container (e.g.,
    running test suites after an agent modifies code). Defaults to ``None``.

    Roadmap: the primary fields will flip at v1.0 — ``item`` and ``sample``
    become authoritative, ``response`` / ``target`` / ``metadata`` / ``config``
    become legacy accessors. Existing downstream code keeps working through
    deprecation; see the integration RFC for the migration plan.
    """

    response: str
    target: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)
    sandbox: Sandbox | None = None

    @property
    def item(self) -> dict[str, Any]:
        """Dataset-row view of the input (NMP-native access).

        Constructed as ``{"reference": self.target, **self.metadata}``.
        NMP metric authors who template with ``{{ item.reference }}`` or
        ``{{ item.<custom_field> }}`` can access their expected shape
        through this property.
        """
        return {"reference": self.target, **self.metadata}

    @property
    def sample(self) -> dict[str, Any]:
        """Inference-payload view of the input (NMP-native access).

        Constructed as ``{"output_text": self.response, "response": self.response}``.
        NMP metric authors who template with ``{{ sample.output_text }}`` or
        read ``sample["output_text"]`` directly get their expected shape.
        """
        return {"output_text": self.response, "response": self.response}

    def jinja_context(self) -> dict[str, Any]:
        """Build a Jinja rendering context supporting both NEL and NMP vocabularies.

        Returned dict includes:

        - Flat keys from :attr:`item` and :attr:`sample` (so
          ``{{ reference }}`` and ``{{ output_text }}`` work)
        - Explicit ``item`` and ``sample`` namespaces (so
          ``{{ item.reference }}`` and ``{{ sample.output_text }}`` work)
        - NEL-native aliases: ``response``, ``target``, ``metadata``,
          ``config`` (so ``{{ response }}`` and ``{{ target }}`` work)

        Use this as the rendering context when rendering templates defined
        on metric classes that need to support both vocabularies::

            from jinja2 import Environment
            ctx = scorer_input.jinja_context()
            rendered = Environment().from_string(template).render(**ctx)
        """
        item = self.item
        sample = self.sample
        return {
            **item,
            **sample,
            "item": item,
            "sample": sample,
            "response": self.response,
            "target": self.target,
            "metadata": self.metadata,
            "config": self.config,
        }
