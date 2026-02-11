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

"""Built-in scorer functions for BYOB benchmarks."""

import re
from collections import Counter


def contains(response: str, target: str, metadata: dict) -> dict:
    """Check if target string is contained in response (case-insensitive)."""
    return {"correct": target.lower().strip() in response.lower()}


def exact_match(response: str, target: str, metadata: dict) -> dict:
    """Check if response exactly matches target (case-insensitive, whitespace-stripped)."""
    return {"correct": response.strip().lower() == target.strip().lower()}


def f1_token(response: str, target: str, metadata: dict) -> dict:
    """Compute token-level F1 score between response and target."""
    pred_tokens = response.lower().split()
    ref_tokens = target.lower().split()

    if not pred_tokens or not ref_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_common = sum(common.values())

    if num_common == 0:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}

    precision = num_common / len(pred_tokens)
    recall = num_common / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)

    return {"f1": f1, "precision": precision, "recall": recall}


def regex_match(response: str, target: str, metadata: dict) -> dict:
    """Check if target regex pattern matches anywhere in response."""
    try:
        return {"correct": bool(re.search(target, response, re.IGNORECASE))}
    except re.error:
        return {"correct": False}


BUILTIN_SCORERS = {
    "contains": contains,
    "exact_match": exact_match,
    "f1_token": f1_token,
    "regex_match": regex_match,
}
