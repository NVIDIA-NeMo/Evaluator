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

"""BoolQ benchmark onboarded as a BYOB example.

BoolQ is a yes/no question answering dataset from Google/SuperGLUE.

Usage:
  python -m nemo_evaluator.byob.cli examples/byob/boolq/benchmark.py
"""
import os

from nemo_evaluator.byob import benchmark, scorer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="boolq",
    dataset=os.path.join(_SCRIPT_DIR, "data.jsonl"),
    prompt=(
        "Read the passage and answer the question with 'yes' or 'no'.\n\n"
        "Passage: {passage}\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def boolq_scorer(response: str, target: str, metadata: dict) -> dict:
    response_lower = response.strip().lower()

    # Extract yes/no from response (priority: startswith > contains > fallback)
    if response_lower.startswith("yes"):
        predicted = "yes"
    elif response_lower.startswith("no"):
        predicted = "no"
    elif "yes" in response_lower and "no" not in response_lower:
        predicted = "yes"
    elif "no" in response_lower and "yes" not in response_lower:
        predicted = "no"
    else:
        predicted = response_lower[:10]  # fallback

    target_str = "yes" if target.lower() in ("true", "yes", "1") else "no"
    return {"correct": predicted == target_str}
