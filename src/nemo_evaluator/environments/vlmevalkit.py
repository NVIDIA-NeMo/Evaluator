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
"""VLMEvalKit datasets as EvalEnvironments (``vlmevalkit://`` URI scheme)."""

from __future__ import annotations

import logging
import re
import string
from typing import TYPE_CHECKING, Any

import pandas as pd

from nemo_evaluator.environments.base import EvalEnvironment, SeedResult, VerifyResult

if TYPE_CHECKING:
    from nemo_evaluator.sandbox.base import Sandbox

logger = logging.getLogger(__name__)


def _ensure_importable():
    try:
        import vlmeval  # noqa: F401
    except ImportError:
        raise ImportError(
            "VLMEvalKit not found. Install it or add it to PYTHONPATH:\n"
            "  pip install vlmeval\n"
            "  # or: git clone https://github.com/open-compass/VLMEvalKit && "
            "export PYTHONPATH=$PYTHONPATH:$(pwd)/VLMEvalKit"
        )


def _extract_mcq_answer(response: str, choices: dict[str, str]) -> str | None:
    """Extract a single option letter from the response (mirrors VLMEvalKit ``can_infer``)."""
    try:
        from vlmeval.utils.matching_util import can_infer

        result = can_infer(response, choices)
        if result and result != "Z":
            return result
    except Exception:
        pass

    clean = response.strip()
    for ch in ".()[],:;!*#{}":
        clean = clean.replace(ch, " ")
    tokens = clean.split()

    valid = set(choices.keys())
    found = [t for t in tokens if t in valid]
    if len(found) == 1:
        return found[0]

    answer_match = re.search(r"(?i)(?:answer|choice|option)\s*(?:is|:)?\s*([A-Z])", response)
    if answer_match and answer_match.group(1) in valid:
        return answer_match.group(1)

    return None


class VLMEvalKitEnvironment(EvalEnvironment):
    """VLMEvalKit dataset wrapper. Supports MCQ, VQA, and Y/N types."""

    def __init__(
        self,
        dataset_name: str,
        limit: int | None = None,
    ) -> None:
        super().__init__()
        _ensure_importable()
        from vlmeval.dataset import (
            DATASET_TYPE,
            SUPPORTED_DATASETS,
            build_dataset,
        )

        if dataset_name not in SUPPORTED_DATASETS:
            available = ", ".join(sorted(SUPPORTED_DATASETS)[:20])
            raise ValueError(f"Unknown VLMEvalKit dataset {dataset_name!r}. Available (first 20): {available} ...")

        self.dataset_name = dataset_name
        self.name = f"vlmevalkit://{dataset_name}"
        self._dataset_type = DATASET_TYPE(dataset_name)

        logger.info("Loading VLMEvalKit dataset %s (type=%s)...", dataset_name, self._dataset_type)
        self._vk_dataset = build_dataset(dataset_name)
        self._data: pd.DataFrame = self._vk_dataset.data

        if limit is not None and limit < len(self._data):
            self._data = self._data.iloc[:limit].reset_index(drop=True)

        logger.info(
            "VLMEvalKit %s: %d samples, type=%s",
            dataset_name,
            len(self._data),
            self._dataset_type,
        )

    async def dataset_size(self) -> int:
        return len(self._data)

    async def seed(self, idx: int) -> SeedResult:
        line = self._data.iloc[idx]
        line_dict = dict(line)

        msgs = self._vk_dataset.build_prompt(line)

        images: list[str] = []
        prompt_text = ""
        for msg in msgs:
            if msg["type"] == "image":
                val = msg["value"]
                if isinstance(val, list):
                    images.extend(val)
                else:
                    images.append(val)
            elif msg["type"] == "text":
                prompt_text = msg["value"]

        expected = str(line_dict.get("answer", ""))

        choices = {}
        for letter in string.ascii_uppercase:
            if letter in line_dict and not pd.isna(line_dict[letter]):
                choices[letter] = str(line_dict[letter])

        category = ""
        for cat_key in ("category", "l2-category", "l2_category", "split"):
            if cat_key in line_dict and not pd.isna(line_dict.get(cat_key)):
                category = str(line_dict[cat_key])
                break

        return SeedResult(
            prompt=prompt_text,
            expected_answer=expected,
            images=images or None,
            metadata={
                "source": "vlmevalkit",
                "dataset": self.dataset_name,
                "dataset_type": self._dataset_type,
                "index": str(line_dict.get("index", idx)),
                "category": category,
                "choices": choices,
            },
        )

    async def verify(
        self,
        response: str,
        expected: str,
        sandbox: Sandbox | None = None,
        **meta: Any,
    ) -> VerifyResult:
        dataset_type = meta.get("dataset_type", self._dataset_type)
        choices = meta.get("choices", {})

        if dataset_type == "MCQ":
            return self._score_mcq(response, expected, choices)
        elif dataset_type == "Y/N":
            return self._score_yorn(response, expected)
        else:
            return self._score_vqa(response, expected)

    def _score_mcq(self, response: str, expected: str, choices: dict[str, str]) -> VerifyResult:
        predicted = _extract_mcq_answer(response, choices)
        correct = predicted is not None and predicted.upper() == expected.upper()
        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=predicted or response[:200],
            scoring_details={
                "method": "vlmevalkit_mcq",
                "predicted_option": predicted,
                "expected_option": expected,
                "exact_match": correct,
            },
        )

    def _score_yorn(self, response: str, expected: str) -> VerifyResult:
        resp_lower = response.strip().lower()
        pred = None
        for token in ("yes", "no"):
            if token in resp_lower:
                pred = token
                break
        exp_lower = expected.strip().lower()
        correct = pred is not None and pred == exp_lower
        return VerifyResult(
            reward=1.0 if correct else 0.0,
            extracted_answer=pred or response[:200],
            scoring_details={
                "method": "vlmevalkit_yorn",
                "predicted": pred,
                "expected": exp_lower,
            },
        )

    def _score_vqa(self, response: str, expected: str) -> VerifyResult:
        pred = response.strip().lower()
        exp = expected.strip().lower()
        exact = pred == exp
        contains = exp in pred if exp else False
        reward = 1.0 if exact else (0.5 if contains else 0.0)
        return VerifyResult(
            reward=reward,
            extracted_answer=response[:500],
            scoring_details={
                "method": "vlmevalkit_vqa",
                "exact_match": exact,
                "contains_match": contains,
            },
        )
