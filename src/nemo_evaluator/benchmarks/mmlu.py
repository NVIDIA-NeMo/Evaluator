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
"""MMLU -- Massive Multitask Language Understanding (14K 4-choice QA)."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, multichoice_regex

_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _prepare(row, idx, rng):
    c = row["choices"]
    return {
        **row,
        "Question": row["question"],
        "A": c[0],
        "B": c[1],
        "C": c[2],
        "D": c[3],
        "answer": "ABCD"[row["answer"]],
        "category": row.get("subject", ""),
    }


@benchmark(
    name="mmlu",
    dataset="hf://cais/mmlu?config=all&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def mmlu_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
