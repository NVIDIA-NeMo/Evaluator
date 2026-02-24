# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MT-Bench: multi-turn conversation benchmark with LLM-as-Judge.

80 samples, 2 turns each. Tests conversation quality across 8 categories
(writing, roleplay, reasoning, math, coding, extraction, stem, humanities).

Uses MultiTurnStrategy for multi-turn model interaction, then LLM-as-Judge
to score conversation quality on a 1-10 scale.

Source: HuggingFaceH4/mt_bench_prompts on HuggingFace (train split).

Usage:
  nemo-evaluator-byob examples/byob/mt_bench/benchmark.py
  nemo-evaluator run_eval --eval_type byob_mt_bench.mt-bench ...
"""

from nemo_evaluator.byob import benchmark, scorer, ScorerInput
from nemo_evaluator.byob.judge import (
    JudgeConfig,
    judge_call,
    parse_grade,
    render_judge_prompt,
    _get_judge_session,
)

MT_BENCH_JUDGE_TEMPLATE = """\
Please act as an impartial judge and evaluate the quality of the response \
provided by an AI assistant to the user question displayed below. You will \
be given the full multi-turn conversation. Evaluate the assistant's helpfulness, \
relevance, accuracy, depth, creativity, and level of detail.

[Conversation]
{conversation}

[End of Conversation]

{criteria}

Rate the overall quality of the assistant's responses on a scale of 1 to 10, \
where 1 is extremely poor and 10 is excellent.

Think step by step, then output your rating on a new line:
GRADE: [[score]]
where score is an integer from 1 to 10.\
"""


def _format_conversation(conversation):
    """Format conversation list into readable text for the judge."""
    lines = []
    for msg in conversation:
        role = msg.get("role", "unknown").capitalize()
        content = msg.get("content", "")
        lines.append(f"**{role}:** {content}")
    return "\n\n".join(lines)


@benchmark(
    name="mt-bench",
    dataset="hf://HuggingFaceH4/mt_bench_prompts?split=train",
    # prompt is ignored by MultiTurnStrategy (it reads row["prompt"] directly)
    prompt="{prompt}",
    target_field="reference",
    endpoint_type="chat",
    requirements=["datasets"],
    extra={
        "judge": {
            "url": "https://inference-api.nvidia.com/v1",
            "model_id": "gcp/google/gemini-2.5-flash",
            "api_key": "JUDGE_API_KEY",
            "temperature": 0.0,
            "max_new_tokens": 2048,
        },
        # Signal to the runner that this uses multi-turn strategy
        "strategy": "multi_turn",
        "turns_field": "prompt",
    },
)
@scorer
def mt_bench_scorer(sample: ScorerInput) -> dict:
    """Score the multi-turn conversation using LLM-as-Judge (1-10 scale)."""
    # Format the full conversation for the judge
    conversation = sample.conversation or []
    conv_text = _format_conversation(conversation)

    # Resolve judge config
    judge_cfg = sample.config.get("judge")
    if judge_cfg is None:
        return {"quality": 0.0, "judge_grade": "NO_CONFIG"}

    config = JudgeConfig.from_dict(judge_cfg) if isinstance(judge_cfg, dict) else judge_cfg

    category = sample.metadata.get("category", "general")
    prompt = render_judge_prompt(
        MT_BENCH_JUDGE_TEMPLATE,
        conversation=conv_text,
        criteria=f"Category: {category}. Pay special attention to the follow-up response quality.",
    )

    try:
        session = sample.config.get("_judge_session") or _get_judge_session(config)
        judge_response = judge_call(config, prompt, session=session)
    except Exception:
        return {"quality": 0.0, "judge_grade": "CALL_ERROR"}

    # Parse 1-10 score (MT-Bench uses [[score]] format)
    grade = parse_grade(judge_response, r"\[\[(\d+)\]\]")
    if grade is None:
        # Fallback: try GRADE: N format
        grade = parse_grade(judge_response, r"GRADE:\s*\[?\[?(\d+)\]?\]?")
    if grade is None:
        return {"quality": 0.0, "judge_grade": "PARSE_ERROR"}

    try:
        score = float(grade)
        # Normalize 1-10 to 0-1
        normalized = max(0.0, min(1.0, (score - 1) / 9))
    except ValueError:
        return {"quality": 0.0, "judge_grade": "PARSE_ERROR"}

    scores = {
        "quality": normalized,
        "raw_score": score,
        "judge_grade": grade,
    }

    # Per-category breakdown
    if category:
        scores[f"quality_{category}"] = normalized

    return scores
