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
"""DROP -- reading comprehension requiring discrete reasoning (few-shot)."""

import random

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, fuzzy_match

_HEADER = (
    "You will be asked to read a passage and answer a question. Some examples of passages and Q&A are provided below."
)
_FOOTER = '\nThink step by step, then write a line of the form "Answer: $ANSWER" at the end of your response.\n'

# DROP data comes from HF but needs special parsing
_TRAIN_CACHE: list[dict] = []
_TEST_CACHE: list[dict] = []


def _load_drop():
    if _TEST_CACHE:
        return
    from datasets import load_dataset

    ds = load_dataset("ucinlp/drop", split="validation")
    train_ds = load_dataset("ucinlp/drop", split="train")

    for row in train_ds:
        _TRAIN_CACHE.append(
            {
                "context": row["passage"],
                "completion": row["answers_spans"]["spans"][0] if row["answers_spans"]["spans"] else "",
            }
        )

    for row in ds:
        answers = list(row["answers_spans"]["spans"]) if row["answers_spans"]["spans"] else [""]
        _TEST_CACHE.append(
            {
                "passage": row["passage"],
                "question": row["question"],
                "answers": answers,
            }
        )


def _seed_drop(row, idx):
    """Custom seed: build few-shot prompt with deterministic examples."""
    rng = random.Random(42 + idx)
    stuffing = rng.sample(_TRAIN_CACHE, min(3, len(_TRAIN_CACHE)))
    parts = [_HEADER, "\n\n# Examples"]
    for s in stuffing:
        parts.append(f"\n---\n{s['context']} {s['completion']}\n")
    parts.append("\n# Your Task\n")
    parts.append(f"\n---\n{row['passage']} {row['question']}")
    parts.append(_FOOTER)
    prompt = "".join(parts)
    return SeedResult(
        prompt=prompt,
        expected_answer=row["answers"][0],
        messages=[{"role": "user", "content": prompt}],
        metadata={"correct_answers": row["answers"]},
    )


def _load_drop_data():
    _load_drop()
    return list(_TEST_CACHE)


@benchmark(name="drop", dataset=_load_drop_data, target_field="answers", prompt="", seed_fn=_seed_drop)
@scorer
def drop_scorer(sample: ScorerInput) -> dict:
    return fuzzy_match(sample)
