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
"""MMLU-Pro -- 10-choice version from TIGER-Lab."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, multichoice_regex

_LETTERS = "ABCDEFGHIJ"
_PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be in the following format: 'Answer: $LETTER' where LETTER is one of "
    + "/".join(_LETTERS)
    + ".\n\n{Question}\n\n"
    + "\n".join(f"{L}) {{{L}}}" for L in _LETTERS)
)


def _prepare(row, idx, rng):
    opts = row["options"]
    while len(opts) < 10:
        opts.append("")
    return {
        **row,
        "Question": row["question"],
        **{L: opts[i] for i, L in enumerate(_LETTERS[: len(opts)])},
        "category": row.get("category", ""),
    }


@benchmark(
    name="mmlu_pro",
    dataset="hf://TIGER-Lab/MMLU-Pro?split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def mmlu_pro_scorer(sample: ScorerInput) -> dict:
    return multichoice_regex(sample, pattern=r"(?i)Answer\s*:\s*([A-J])")
