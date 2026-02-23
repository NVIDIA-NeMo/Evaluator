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

"""Download MedMCQA validation split and create a stratified sample.

This script is NOT needed at runtime -- it generates data.jsonl which is
committed to the repo. Run it once when refreshing the dataset.

Requirements:
    pip install datasets

Usage:
    python examples/byob/medmcqa/prepare_data.py
"""

import json
import os
import random

from datasets import load_dataset

SEED = 42
TOTAL_SAMPLES = 200
COP_TO_LETTER = {0: "A", 1: "B", 2: "C", 3: "D"}
OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data.jsonl")


def main():
    random.seed(SEED)

    print("Loading MedMCQA validation split...")
    ds = load_dataset("openlifescienceai/medmcqa", split="validation")

    # Filter to single-choice questions only
    ds = ds.filter(lambda x: x["choice_type"] == "single")
    print(f"Single-choice questions: {len(ds)}")

    # Group by subject_name for stratified sampling
    by_subject = {}
    for row in ds:
        subj = row["subject_name"]
        by_subject.setdefault(subj, []).append(row)

    print(f"Subjects found: {len(by_subject)}")

    # Proportional stratified sampling
    total_available = sum(len(v) for v in by_subject.values())
    sampled = []
    remaining = TOTAL_SAMPLES

    subjects_sorted = sorted(by_subject.keys())
    for i, subj in enumerate(subjects_sorted):
        rows = by_subject[subj]
        if i == len(subjects_sorted) - 1:
            # Last subject gets the remainder to hit exact total
            n = remaining
        else:
            n = max(1, round(len(rows) / total_available * TOTAL_SAMPLES))
            n = min(n, remaining)

        random.shuffle(rows)
        sampled.extend(rows[:n])
        remaining -= min(n, len(rows))
        print(f"  {subj}: {min(n, len(rows))} samples (from {len(rows)} available)")

    print(f"\nTotal sampled: {len(sampled)}")

    # Shuffle final dataset
    random.shuffle(sampled)

    # Write JSONL
    with open(OUTPUT_PATH, "w") as f:
        for row in sampled:
            record = {
                "question": row["question"],
                "a": row["opa"],
                "b": row["opb"],
                "c": row["opc"],
                "d": row["opd"],
                "answer": COP_TO_LETTER[row["cop"]],
                "subject_name": row["subject_name"],
                "topic_name": row.get("topic_name") or "",
                "exp": row.get("exp") or "",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Written {len(sampled)} samples to {OUTPUT_PATH}")

    # Spot-check: verify cop mapping against explanations
    print("\nSpot-check (first 3 samples):")
    with open(OUTPUT_PATH) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            rec = json.loads(line)
            print(f"  Q: {rec['question'][:80]}...")
            exp = rec['exp'] or ""
            print(f"  Answer: {rec['answer']} | Explanation: {exp[:100]}")
            print()


if __name__ == "__main__":
    main()
