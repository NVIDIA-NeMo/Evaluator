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
"""MATH-500 -- competition math from HuggingFaceH4."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, answer_line

_PROMPT = (
    "Solve the following math problem step by step. The last line of your response "
    "should be of the form Answer: $ANSWER (without quotes) where $ANSWER is the answer "
    "to the problem.\n\n{problem}"
)


def _prepare(row, idx, rng):
    return {**row, "category": row.get("type", ""), "level": row.get("level", "")}


@benchmark(
    name="math500",
    dataset="hf://HuggingFaceH4/MATH-500?split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def math500_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample)
