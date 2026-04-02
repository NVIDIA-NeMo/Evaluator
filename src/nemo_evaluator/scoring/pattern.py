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
"""Pattern-based scorers: regex extraction and numeric matching."""

from __future__ import annotations

import re

from nemo_evaluator.scoring.types import ScorerInput


def multichoice_regex(sample: ScorerInput, pattern: str = r"(?i)Answer\s*:\s*([A-D])") -> dict:
    m = re.search(pattern, sample.response)
    extracted = m.group(1).upper() if m else None
    return {"correct": extracted == str(sample.target).upper(), "extracted": extracted}


def answer_line(sample: ScorerInput, pattern: str = r"(?i)Answer\s*:\s*([^\n]+)") -> dict:
    m = re.search(pattern, sample.response)
    extracted = m.group(1).strip() if m else sample.response.strip().split("\n")[-1]
    return {
        "correct": _normalize_math(extracted) == _normalize_math(str(sample.target)),
        "extracted": extracted,
    }


def numeric_match(sample: ScorerInput) -> dict:
    nums = re.findall(r"-?\d+\.?\d*", sample.response.replace(",", ""))
    extracted = nums[-1].rstrip("0").rstrip(".") if nums else ""
    target = str(sample.target).strip().rstrip("0").rstrip(".")
    return {"correct": extracted == target, "extracted": extracted}


def _normalize_math(s: str) -> str:
    s = s.strip().lower()
    for ch in (",", "$", "%", " "):
        s = s.replace(ch, "")
    return s.rstrip(".")
