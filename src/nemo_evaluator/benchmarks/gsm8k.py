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
"""GSM8K -- grade school math (1.3K test problems)."""

import re
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, answer_line

_PROMPT = (
    "Solve the following math problem step by step. "
    "Put your final numerical answer after 'The answer is'.\n\n"
    "Problem: {question}"
)


def _prepare(row, idx, rng):
    m = re.search(r"####\s*(.+)", row.get("answer", ""))
    answer = m.group(1).strip().replace(",", "") if m else row.get("answer", "")
    return {**row, "answer": answer}


@benchmark(
    name="gsm8k",
    dataset="hf://openai/gsm8k?config=main&split=test",
    prompt=_PROMPT,
    target_field="answer",
    prepare_row=_prepare,
)
@scorer
def gsm8k_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample, pattern=r"(?i)(?:the answer is|answer\s*:)\s*([^\n]+)")
