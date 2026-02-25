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

"""Open-ended question answering benchmark template.

Suitable for: TriviaQA, NaturalQuestions, reading comprehension, RAG evaluation.
Scoring: Check if target answer appears in response (contains), or use F1 token overlap.

Dataset fields: question (str), context (str, optional), answer (str)
Target field: answer
"""
import os

from nemo_evaluator.byob import benchmark, scorer
from nemo_evaluator.byob.scorers import contains

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="open_qa",
    dataset=os.path.join(_SCRIPT_DIR, "open_qa_data.jsonl"),
    prompt=(
        "Answer the following question. Be concise.\n\n"
        "Question: {question}\n\n"
        "Answer:"
    ),
    target_field="answer",
    endpoint_type="chat",
)
@scorer
def qa_scorer(response: str, target: str, metadata: dict) -> dict:
    """Check if target answer is contained in model response.

    Uses the built-in contains scorer for simplicity.
    For stricter matching, replace with exact_match or f1_token.
    """
    return contains(response, target, metadata)
