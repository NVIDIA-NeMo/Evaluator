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

"""MedMCQA benchmark onboarded as a BYOB example.

MedMCQA is a large-scale multiple-choice QA dataset from Indian medical
entrance exams (AIIMS & NEET PG), covering 21 medical subjects with ~194K
questions. Dataset is downloaded from HuggingFace at runtime.

Source: openlifescienceai/medmcqa on HuggingFace (validation split).

Dataset fields (HF): question, opa, opb, opc, opd, cop (0-3), subject_name,
                     topic_name, exp (explanation)

Usage:
  python -m nemo_evaluator.contrib.byob.cli examples/byob/medmcqa/benchmark.py
"""

import re

from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

# Map HF integer answer codes to letters
_COP_TO_LETTER = {"0": "A", "1": "B", "2": "C", "3": "D"}


@benchmark(
    name="medmcqa",
    dataset="hf://openlifescienceai/medmcqa?split=validation",
    prompt=(
        "You are a medical expert taking a licensing examination.\n\n"
        "Question: {question}\n\n"
        "A) {a}\n"
        "B) {b}\n"
        "C) {c}\n"
        "D) {d}\n\n"
        "Answer with just the letter (A, B, C, or D):"
    ),
    target_field="cop",
    endpoint_type="chat",
    requirements=["datasets"],
    field_mapping={"opa": "a", "opb": "b", "opc": "c", "opd": "d"},
)
@scorer
def medmcqa_scorer(sample: ScorerInput) -> dict:
    """Extract letter choice from response and compare to target.

    The HF dataset stores the correct answer as an integer (cop: 0-3).
    We convert it to a letter (A-D) for comparison.

    Handles response formats like:
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
            r"(?:answer\s+is\s+|^\s*\(?)\s*([A-Da-d])\b",
            response_clean,
            re.IGNORECASE,
        )
        if match:
            predicted = match.group(1).upper()
        else:
            # Last resort: find any standalone A-D in first 50 chars
            match = re.search(r"\b([A-Da-d])\b", response_clean[:50])
            predicted = match.group(1).upper() if match else ""

    # Convert HF integer target (0-3) to letter (A-D)
    target_str = str(sample.target).strip()
    target_letter = _COP_TO_LETTER.get(target_str, target_str.upper())

    return {
        "correct": predicted == target_letter,
        "parsed": bool(predicted),
    }
