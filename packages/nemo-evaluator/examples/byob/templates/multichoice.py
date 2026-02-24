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

"""Multiple choice benchmark template.

Suitable for: MMLU, ARC, HellaSwag, MedQA, custom MC datasets.
Scoring: Extract letter answer (A/B/C/D) from response, compare to target.

Dataset fields: question (str), a (str), b (str), c (str), d (str), answer (str: "A"/"B"/"C"/"D")
Target field: answer
"""
import os

from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="multichoice",
    dataset=os.path.join(_SCRIPT_DIR, "multichoice_data.jsonl"),
    prompt=(
        "{question}\n\n"
        "A) {a}\n"
        "B) {b}\n"
        "C) {c}\n"
        "D) {d}\n\n"
        "Answer with just the letter (A, B, C, or D):"
    ),
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def multichoice_scorer(sample: ScorerInput) -> dict:
    """Extract letter choice from response.

    Handles formats like:
    - "A"
    - "A)"
    - "The answer is B"
    - "B. Because..."
    - "(C)"
    """
    import re

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
            # Last resort: find any standalone A-D
            match = re.search(r'\b([A-Da-d])\b', response_clean[:50])
            predicted = match.group(1).upper() if match else ""

    return {
        "correct": predicted == sample.target.strip().upper(),
        "parsed": bool(predicted),
    }
