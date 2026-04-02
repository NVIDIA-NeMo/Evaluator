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
"""TriviaQA -- trivia questions with multi-alias exact match."""

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput


def _prepare(row, idx, rng):
    aliases = list(
        {row["answer"]["value"]}
        | set(row["answer"].get("aliases", []))
        | set(row["answer"].get("normalized_aliases", [])) - {""}
    )
    return {**row, "question": row["question"], "answer_text": row["answer"]["value"], "_aliases": aliases}


def _load_triviaqa():
    from datasets import load_dataset

    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")
    return [dict(row) for row in ds]


@benchmark(
    name="triviaqa",
    dataset=_load_triviaqa,
    prompt="Answer with a short factual answer.\n\nQuestion: {question}\nAnswer:",
    target_field="answer_text",
    prepare_row=_prepare,
)
@scorer
def triviaqa_scorer(sample: ScorerInput) -> dict:
    first_line = sample.response.strip().split("\n")[0].strip()
    aliases = sample.metadata.get("_aliases", [str(sample.target)])
    from nemo_evaluator.scoring.text import _normalize

    pred = _normalize(first_line)
    correct = any(_normalize(a) == pred for a in aliases)
    return {"correct": correct, "extracted": first_line}
