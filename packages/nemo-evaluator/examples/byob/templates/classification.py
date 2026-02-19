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

"""Text classification benchmark template.

Suitable for: Sentiment analysis, topic classification, intent detection.
Scoring: Check if predicted label matches target label (case-insensitive).

Dataset fields: text (str), label (str), categories (str)
Target field: label
"""
import os

from nemo_evaluator.byob import benchmark, scorer

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


@benchmark(
    name="classification",
    dataset=os.path.join(_SCRIPT_DIR, "classification_data.jsonl"),
    prompt=(
        "Classify the following text into exactly one category.\n"
        "Categories: {categories}\n\n"
        "Text: {text}\n\n"
        "Category:"
    ),
    target_field="label",
    endpoint_type="chat",
)
@scorer
def classification_scorer(response: str, target: str, metadata: dict) -> dict:
    """Extract classification label from first line of response."""
    predicted = response.strip().split('\n')[0].strip().lower()
    target_lower = target.strip().lower()

    # Exact match on first line
    exact = predicted == target_lower
    # Partial: target label appears anywhere in first line
    partial = target_lower in predicted

    return {"correct": exact or partial, "exact": exact}
