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

"""Global-MMLU-Lite benchmark onboarded as a BYOB example.

Global-MMLU-Lite is a lightweight, multilingual version of MMLU from
Cohere Labs, designed to evaluate knowledge and reasoning across 18
languages and diverse cultural contexts. Dataset is downloaded from
HuggingFace at runtime.

Source: CohereForAI/Global-MMLU-Lite on HuggingFace (English test split).

Dataset fields (HF): question, option_a, option_b, option_c, option_d,
                     answer (A/B/C/D), subject, subject_category,
                     cultural_sensitivity_label

Usage:
  python -m nemo_evaluator.byob.cli examples/byob/global_mmlu_lite/benchmark.py
"""
import re

from nemo_evaluator.byob import benchmark, scorer, ScorerInput


@benchmark(
    name="global-mmlu-lite",
    dataset="hf://CohereForAI/Global-MMLU-Lite/en?split=test",
    prompt=(
        "The following is a multiple choice question about {subject}.\n\n"
        "{question}\n\n"
        "A) {a}\n"
        "B) {b}\n"
        "C) {c}\n"
        "D) {d}\n\n"
        "Answer with just the letter (A, B, C, or D):"
    ),
    target_field="answer",
    endpoint_type="chat",
    requirements=["datasets"],
    field_mapping={
        "option_a": "a",
        "option_b": "b",
        "option_c": "c",
        "option_d": "d",
        "cultural_sensitivity_label": "cultural_sensitivity",
    },
)
@scorer
def global_mmlu_lite_scorer(sample: ScorerInput) -> dict:
    """Extract letter choice from response and compare to target.

    Handles formats like:
    - "A"
    - "A)"
    - "The answer is B"
    - "B. Because..."
    - "(C)"
    """
    response_clean = sample.response.strip()

    # Try: first character is A-D
    if response_clean and response_clean[0].upper() in "ABCD":
        predicted = response_clean[0].upper()
    else:
        # Try: find "answer is X" or standalone letter
        match = re.search(
            r'(?:answer\s+is\s+|^\s*\(?)\s*([A-Da-d])\b',
            response_clean,
            re.IGNORECASE,
        )
        if match:
            predicted = match.group(1).upper()
        else:
            # Last resort: find any standalone A-D in first 50 chars
            match = re.search(r'\b([A-Da-d])\b', response_clean[:50])
            predicted = match.group(1).upper() if match else ""

    is_correct = predicted == sample.target.strip().upper()
    is_parsed = bool(predicted)

    scores = {
        "correct": is_correct,
        "parsed": is_parsed,
    }

    # Per-category breakdown (e.g., correct_STEM, correct_Medical)
    category = sample.metadata.get("subject_category")
    if category:
        scores[f"correct_{category}"] = is_correct

    return scores
