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
"""MGSM -- Multilingual Grade School Math (11 languages)."""

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, numeric_match

_LANGS = ["en", "de", "fr", "es", "ru", "zh", "ja", "th", "te", "bn", "sw"]

_INSTRUCTIONS = {
    "en": "Solve this math problem. Give reasoning steps then the final answer on the last line as 'Answer:'.\n\n{input}",
    "de": "Lose dieses Problem. Gib Schritte an, dann die Antwort als 'Antwort:'.\n\n{input}",
    "fr": "Resolvez ce probleme. Donnez les etapes puis la reponse comme 'Reponse:'.\n\n{input}",
    "es": "Resuelve este problema. Da los pasos y la respuesta como 'Respuesta:'.\n\n{input}",
    "ru": "Reshite etu zadachu. Privedite shagi, zatem otvet kak 'Otvet:'.\n\n{input}",
    "zh": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "ja": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "th": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "te": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "bn": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
    "sw": "Solve this math problem step by step. Final answer as 'Answer:'.\n\n{input}",
}


def _load_mgsm():
    from datasets import load_dataset

    rows = []
    for lang in _LANGS:
        ds = load_dataset("juletxara/mgsm", lang, split="test", trust_remote_code=True)
        for row in ds:
            answer = str(row["answer_number"]).rstrip("0").rstrip(".")
            rows.append({"question": row["question"], "answer": answer, "language": lang})
    return rows


def _seed_mgsm(row, idx):
    lang = row["language"]
    template = _INSTRUCTIONS.get(lang, _INSTRUCTIONS["en"])
    prompt = template.format(input=row["question"])
    return SeedResult(
        prompt=prompt,
        expected_answer=row["answer"],
        messages=[{"role": "user", "content": prompt}],
        metadata={"category": lang},
    )


@benchmark(name="mgsm", dataset=_load_mgsm, target_field="answer", prompt="", seed_fn=_seed_mgsm)
@scorer
def mgsm_scorer(sample: ScorerInput) -> dict:
    return numeric_match(sample)
