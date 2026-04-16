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
#!/usr/bin/env python3
"""Download the full XSTest v2 dataset (450 prompts) and convert to nemo-gym JSONL format.

Source: paul-rottger/exaggerated-safety (CC-BY-4.0)
Paper: https://arxiv.org/abs/2308.01263

Usage:
    python -m nemo_evaluator.resources.xstest_data <output_path>

Skips download if the output file already exists.
"""

from __future__ import annotations

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

XSTEST_CSV_URL = "https://raw.githubusercontent.com/paul-rottger/exaggerated-safety/main/xstest_v2_prompts.csv"

EXPECTED_PROMPTS = 450


def download_and_convert(output_path: str) -> None:
    if Path(output_path).exists():
        count = sum(1 for _ in open(output_path))
        print(f"[xstest_data] {output_path} already exists ({count} rows), skipping")
        return

    print(f"[xstest_data] Downloading XSTest v2 from {XSTEST_CSV_URL}")
    data = urllib.request.urlopen(XSTEST_CSV_URL).read().decode("utf-8")
    reader = csv.DictReader(io.StringIO(data))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for row in reader:
            record = {
                "responses_create_params": {
                    "input": [{"role": "user", "content": row["prompt"]}],
                },
                "verifier_metadata": {
                    "id": int(row["id_v2"]),
                    "type": row["type"],
                    "label": row["label"],
                    "focus": row.get("focus", ""),
                    "note": row.get("note", ""),
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    print(f"[xstest_data] Wrote {count} prompts to {output_path}")
    if count != EXPECTED_PROMPTS:
        print(f"[xstest_data] WARNING: expected {EXPECTED_PROMPTS} prompts, got {count}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_path>", file=sys.stderr)
        sys.exit(1)
    download_and_convert(sys.argv[1])


if __name__ == "__main__":
    main()
