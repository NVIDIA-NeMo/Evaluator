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

"""Download Global-MMLU-Lite and create evaluation data.

This script is NOT needed at runtime -- it generates data.jsonl which is
committed to the repo. Run it once when refreshing the dataset.

Global-MMLU-Lite: 18 languages, ~615 rows each (400 test + 215 dev).
Source: CohereLabs/Global-MMLU-Lite on HuggingFace.

Requirements:
    pip install datasets

Usage:
    python examples/byob/global_mmlu_lite/prepare_data.py
    python examples/byob/global_mmlu_lite/prepare_data.py --lang fr
    python examples/byob/global_mmlu_lite/prepare_data.py --lang all --max-per-lang 50
"""

import argparse
import json
import os

from datasets import load_dataset

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

LANGUAGES = [
    "ar", "bn", "cy", "de", "en", "es", "fr", "hi", "id", "it",
    "ja", "ko", "my", "pt", "sq", "sw", "yo", "zh",
]


def export_language(lang, max_samples=None):
    """Export test split for a single language to JSONL."""
    print(f"Loading {lang}...")
    ds = load_dataset("CohereLabs/Global-MMLU-Lite", lang, split="test")

    records = []
    for row in ds:
        record = {
            "question": row["question"],
            "a": row["option_a"],
            "b": row["option_b"],
            "c": row["option_c"],
            "d": row["option_d"],
            "answer": row["answer"],
            "subject": row["subject"],
            "subject_category": row["subject_category"],
            "cultural_sensitivity": row["cultural_sensitivity_label"],
            "lang": lang,
        }
        records.append(record)

    if max_samples and len(records) > max_samples:
        import random
        random.seed(42)
        random.shuffle(records)
        records = records[:max_samples]

    return records


def main():
    parser = argparse.ArgumentParser(description="Prepare Global-MMLU-Lite data")
    parser.add_argument(
        "--lang", default="en",
        help="Language code (e.g. 'en', 'fr') or 'all' for all languages",
    )
    parser.add_argument(
        "--max-per-lang", type=int, default=None,
        help="Max samples per language (default: use full test split)",
    )
    args = parser.parse_args()

    langs = LANGUAGES if args.lang == "all" else [args.lang]

    all_records = []
    for lang in langs:
        records = export_language(lang, max_samples=args.max_per_lang)
        all_records.extend(records)
        print(f"  {lang}: {len(records)} samples")

    output_path = os.path.join(OUTPUT_DIR, "data.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWritten {len(all_records)} samples to {output_path}")

    # Summary stats
    subjects = {}
    for r in all_records:
        subjects[r["subject_category"]] = subjects.get(r["subject_category"], 0) + 1
    print("\nBy category:")
    for cat, count in sorted(subjects.items()):
        print(f"  {cat}: {count}")

    if len(langs) > 1:
        lang_counts = {}
        for r in all_records:
            lang_counts[r["lang"]] = lang_counts.get(r["lang"], 0) + 1
        print("\nBy language:")
        for lang, count in sorted(lang_counts.items()):
            print(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
