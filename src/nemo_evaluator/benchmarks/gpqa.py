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
"""GPQA Diamond -- graduate-level science QA with shuffled choices."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, multichoice_regex

_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def _prepare(row, idx, rng):
    choices = [row["Correct Answer"], row["Incorrect Answer 1"], row["Incorrect Answer 2"], row["Incorrect Answer 3"]]
    perm = list(range(4))
    rng.shuffle(perm)
    shuffled = [choices[i] for i in perm]
    correct_letter = "ABCD"[shuffled.index(row["Correct Answer"])]
    return {
        **row,
        "Question": row["Question"],
        "A": shuffled[0],
        "B": shuffled[1],
        "C": shuffled[2],
        "D": shuffled[3],
        "answer": correct_letter,
        "category": row.get("High-level domain", ""),
    }


@benchmark(
    name="gpqa",
    dataset="hf://Idavidrein/gpqa?config=gpqa_diamond&split=train",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def gpqa_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample)
