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
"""Generate golden test fixtures by running 5 samples of each benchmark through NVIDIA API.

Usage:
    export NEMO_API_KEY="sk-..."
    python scripts/generate_fixtures.py

Output is written to tests/fixtures/<benchmark>.json and should be committed to git.
The fixtures include full seed data (prompt, expected_answer, metadata) so tests
can run offline without downloading HF datasets.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("generate_fixtures")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures"
N_SAMPLES = 5

API_URL = os.environ.get("NEMO_MODEL_URL", "https://integrate.api.nvidia.com/v1/chat/completions")
MODEL_ID = os.environ.get("NEMO_MODEL_ID", "azure/openai/gpt-5.2")

BENCHMARKS_WITH_CUSTOM_DATASETS = {
    "drop",
    "triviaqa",
    "mgsm",
    "pinchbench",
    "healthbench",
}
JUDGE_BENCHMARKS = {"simpleqa", "healthbench"}

SKIP_BENCHMARKS: set[str] = set()


async def generate_for_benchmark(bench_name: str, api_key: str) -> list[dict[str, Any]]:
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.engine.model_client import ModelClient

    logger.info("=== %s: loading environment (num_examples=%d) ===", bench_name, N_SAMPLES)
    env = get_environment(bench_name, num_examples=N_SAMPLES)
    ds_size = await env.dataset_size()
    n = min(ds_size, N_SAMPLES)
    logger.info("%s: %d samples available, using %d", bench_name, ds_size, n)

    client = ModelClient(
        base_url=API_URL,
        model=MODEL_ID,
        api_key=api_key,
        temperature=0.0,
        max_tokens=2048,
        max_concurrent=4,
    )

    fixtures: list[dict[str, Any]] = []
    try:
        for idx in range(n):
            seed = await env.seed(idx)
            logger.info("%s[%d]: sending prompt (%d chars)...", bench_name, idx, len(seed.prompt))

            msgs = seed.messages or [{"role": "user", "content": seed.prompt}]
            if seed.system:
                msgs = [{"role": "system", "content": seed.system}] + msgs

            try:
                model_resp = await client.chat(messages=msgs)
                response_text = model_resp.content
                logger.info(
                    "%s[%d]: got response (%d chars, %d tokens)",
                    bench_name,
                    idx,
                    len(response_text),
                    model_resp.total_tokens,
                )
            except Exception as e:
                logger.warning("%s[%d]: API error: %s", bench_name, idx, e)
                response_text = ""

            reward = 0.0
            extracted_answer = None
            scoring_details: dict[str, Any] = {}

            if bench_name not in JUDGE_BENCHMARKS:
                try:
                    vr = await env.verify(
                        response_text,
                        seed.expected_answer,
                        **seed.metadata,
                    )
                    reward = vr.reward
                    extracted_answer = vr.extracted_answer
                    scoring_details = vr.scoring_details
                    logger.info("%s[%d]: reward=%.2f extracted=%s", bench_name, idx, reward, extracted_answer)
                except Exception as e:
                    logger.warning("%s[%d]: verify error: %s", bench_name, idx, e)
                    scoring_details = {"error": str(e)}
            else:
                scoring_details = {"method": "skipped_in_fixture_gen", "reason": "judge required"}

            clean_metadata = {}
            for k, v in seed.metadata.items():
                try:
                    json.dumps(v)
                    clean_metadata[k] = v
                except (TypeError, ValueError):
                    clean_metadata[k] = str(v)

            fixtures.append(
                {
                    "idx": idx,
                    "prompt": seed.prompt,
                    "expected_answer": seed.expected_answer,
                    "metadata": clean_metadata,
                    "messages": seed.messages,
                    "system": seed.system,
                    "model_response": response_text,
                    "reward": reward,
                    "extracted_answer": extracted_answer,
                    "scoring_details": scoring_details,
                }
            )
    finally:
        await client.close()
        if hasattr(env, "close"):
            await env.close()

    return fixtures


async def main():
    api_key = os.environ.get("NEMO_API_KEY")
    if not api_key:
        print("ERROR: Set NEMO_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    import nemo_evaluator.benchmarks  # noqa: F401 -- register all benchmarks

    bench_names = [
        "mmlu",
        "mmlu_pro",
        "gpqa",
        "gsm8k",
        "math500",
        "mgsm",
        "drop",
        "triviaqa",
        "humaneval",
        "simpleqa",
        "healthbench",
        "pinchbench",
    ]

    results: dict[str, list[dict]] = {}
    for name in bench_names:
        if name in SKIP_BENCHMARKS:
            logger.info("Skipping %s (needs external resources)", name)
            continue
        try:
            fixtures = await generate_for_benchmark(name, api_key)
            results[name] = fixtures

            out_path = FIXTURE_DIR / f"{name}.json"
            out_path.write_text(
                json.dumps(fixtures, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            logger.info("Wrote %s (%d samples)", out_path, len(fixtures))
        except Exception as e:
            logger.error("FAILED %s: %s", name, e, exc_info=True)

    total = sum(len(v) for v in results.values())
    logger.info("Done: %d benchmarks, %d total fixtures", len(results), total)

    scored = sum(1 for fixtures in results.values() for f in fixtures if f["reward"] > 0)
    logger.info("Scored correctly: %d / %d", scored, total)


if __name__ == "__main__":
    asyncio.run(main())
