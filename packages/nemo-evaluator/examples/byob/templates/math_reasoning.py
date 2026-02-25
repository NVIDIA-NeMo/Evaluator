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

"""Math reasoning benchmark template.

Suitable for: AIME, GSM8K, MATH, competition math, word problems.
Scoring: Extract final numeric answer from model response, compare to integer/float target.

Dataset fields: problem (str), answer (str or number)
Target field: answer
"""

import os

from nemo_evaluator.contrib.byob import benchmark, scorer, ScorerInput

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="math_reasoning",
    dataset=os.path.join(_SCRIPT_DIR, "math_reasoning_data.jsonl"),
    prompt=(
        "Solve the following math problem step by step. "
        "After your reasoning, state your final answer as a single number.\n\n"
        "Problem: {problem}\n\n"
        "Solution:"
    ),
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def math_scorer(sample: ScorerInput) -> dict:
    """Extract the last number from the response and compare to target.

    Handles formats like:
    - "The answer is 42"
    - "Therefore, 42."
    - "42"
    - "Answer: 42"
    - Negative numbers: "-7"
    - Decimals: "3.14"
    """
    import re

    # Extract all numbers (including negative and decimal)
    numbers = re.findall(r"-?\d+\.?\d*", sample.response)
    if not numbers:
        return {"correct": False, "parsed": False}

    predicted = numbers[-1].rstrip(".")

    # Normalize target (strip whitespace, handle string numbers)
    target_clean = str(sample.target).strip().rstrip(".")

    # Compare as strings first (handles integers)
    if predicted == target_clean:
        return {"correct": True, "parsed": True}

    # Compare as floats (handles "3.0" == "3" cases)
    try:
        return {
            "correct": abs(float(predicted) - float(target_clean)) < 1e-6,
            "parsed": True,
        }
    except ValueError:
        return {"correct": False, "parsed": True}
