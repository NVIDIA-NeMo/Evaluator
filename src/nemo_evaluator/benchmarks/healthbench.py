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
"""HealthBench -- health-related QA (requires LLM-as-judge)."""

from nemo_evaluator.environments.base import SeedResult
from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.scoring import ScorerInput, needs_judge


def _load_healthbench():
    import json as _json
    import urllib.request

    from datasets import load_dataset

    try:
        ds = load_dataset("openai/HealthBench", split="test", trust_remote_code=True)
        return [dict(row) for row in ds]
    except Exception:
        pass

    url = "https://huggingface.co/datasets/openai/HealthBench/resolve/main/data/test.jsonl"
    with urllib.request.urlopen(url, timeout=60) as resp:
        text = resp.read().decode()
    rows = [_json.loads(line) for line in text.strip().splitlines() if line.strip()]
    return rows


def _seed_healthbench(row, idx):
    conv = row.get("prompt", row.get("conversation", []))
    if isinstance(conv, list) and conv:
        prompt = conv[-1].get("content", "") if isinstance(conv[-1], dict) else str(conv[-1])
        messages = [
            {"role": m.get("role", "user"), "content": m.get("content", "")} for m in conv if isinstance(m, dict)
        ]
    else:
        prompt = str(conv)
        messages = [{"role": "user", "content": prompt}]
    rubrics = row.get("rubrics", row.get("criteria", []))
    return SeedResult(
        prompt=prompt,
        expected_answer="",
        messages=messages,
        metadata={"category": row.get("category", ""), "criteria": rubrics},
    )


@benchmark(
    name="healthbench",
    dataset=_load_healthbench,
    target_field="",
    prompt="",
    seed_fn=_seed_healthbench,
    extra={"judge": True},
)
@scorer
def healthbench_scorer(sample: ScorerInput) -> dict:
    return needs_judge(sample)
