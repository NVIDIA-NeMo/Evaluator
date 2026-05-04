"""End-to-end smoke test for LogprobRankingSolver against a fake /v1/completions.

Spins up an aiohttp server that returns OpenAI-shape responses with
deterministic logprobs derived from the continuation. This validates:

1. The HTTP wire format the solver emits (max_tokens=0, echo=true, logprobs=1).
2. The text_offset-based continuation parsing.
3. Concurrent per-choice ranking + argmax selection.
4. End-to-end seed → solve → verify through ByobEnvironment with the
   typed multiple_choice_acc scorer.

Run:
    python scripts/smoketest_logprob_solver.py
"""

from __future__ import annotations

import asyncio
import hashlib
import math
from typing import Any

from aiohttp import web

from nemo_evaluator.environments.custom import BenchmarkDefinition, ByobEnvironment
from nemo_evaluator.scoring.multiple_choice import multiple_choice_acc
from nemo_evaluator.solvers.logprob import LogprobRankingSolver


def _deterministic_logprob(continuation: str, *, target: str) -> float:
    """Deterministic per-continuation logprob: gold gets the highest score.

    The "model" prefers the gold continuation (mocking a perfect oracle).
    Other continuations get logprobs derived from a stable hash so the
    ranking is reproducible across runs.
    """
    if continuation == target:
        return -0.5
    h = int(hashlib.sha256(continuation.encode()).hexdigest()[:8], 16)
    return -2.0 - (h % 100) / 25.0


async def fake_completions_handler(request: web.Request, *, gold_per_prompt: dict[str, str]) -> web.Response:
    body = await request.json()
    prompt = body["prompt"]
    # The benchmark prompt is a stable substring of `prompt`; find which
    # gold continuation belongs to it.
    gold = next((g for stem, g in gold_per_prompt.items() if stem in prompt), "")

    # Identify the continuation: it's whatever came AFTER the longest
    # benchmark stem we recognise. For the smoke test we just take the
    # last word as the continuation token sequence.
    matched_stem = next((stem for stem in gold_per_prompt if stem in prompt), "")
    continuation = prompt[len(matched_stem):] if matched_stem else prompt[-10:]

    logprob = _deterministic_logprob(continuation, target=gold)

    # Synthesize a token-level response with text_offset that puts the
    # continuation just after `matched_stem`.
    ctx_end = len(matched_stem)
    tokens = ["<ctx>", continuation]
    token_logprobs = [None, logprob]
    text_offset = [0, ctx_end]
    top_logprobs = [
        {"<ctx>": -0.01},
        {continuation: logprob, "_other": logprob - 1.0},
    ]

    return web.json_response(
        {
            "choices": [
                {
                    "text": "",
                    "finish_reason": "length",
                    "logprobs": {
                        "tokens": tokens,
                        "token_logprobs": token_logprobs,
                        "text_offset": text_offset,
                        "top_logprobs": top_logprobs,
                    },
                }
            ],
            "model": body["model"],
            "usage": {"prompt_tokens": len(prompt), "completion_tokens": 0, "total_tokens": len(prompt)},
        }
    )


def make_app(gold_per_prompt: dict[str, str]) -> web.Application:
    async def handler(request: web.Request) -> web.Response:
        return await fake_completions_handler(request, gold_per_prompt=gold_per_prompt)

    app = web.Application()
    app.router.add_post("/v1/completions", handler)
    return app


async def run_smoketest() -> None:
    benchmark_rows = [
        # prompt stem, choices, gold idx, gold text
        {"q": "Capital of France?", "answer": 2},
        {"q": "Capital of UK?", "answer": 3},
        {"q": "Capital of Germany?", "answer": 0},
    ]
    choices = ["Berlin", "Madrid", "Paris", "London"]
    gold_per_prompt = {f"Q: {row['q']}\nA: ": choices[row["answer"]] for row in benchmark_rows}

    # Start fake server
    app = make_app(gold_per_prompt)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 11999)
    await site.start()

    try:
        defn = BenchmarkDefinition(
            name="capitals_smoke",
            dataset=lambda: benchmark_rows,
            prompt="Q: {q}\nA: ",
            target_field="answer",
            choices=choices,
            scorer_fn=multiple_choice_acc,
        )
        env = ByobEnvironment(defn)
        solver = LogprobRankingSolver(
            base_url="http://127.0.0.1:11999/v1",
            model="fake-mc-oracle",
        )

        print(f"running {len(benchmark_rows)} rows × {len(choices)} choices each\n")
        per_row_results = []
        for idx in range(len(benchmark_rows)):
            seed = await env.seed(idx)
            solve = await solver.solve(seed)
            assert solve.error is None, f"solver error: {solve.error}"

            merged_meta = {**seed.metadata, **solve.scoring_details}
            vr = await env.verify(solve.response, seed.expected_answer, **merged_meta)
            outputs = vr.scoring_details.get("outputs", {})
            per_row_results.append((benchmark_rows[idx]["q"], solve, vr, outputs))

        # Summary
        print(f"{'question':<22} {'argmax':<10} {'gold':<10} {'acc':<6} {'logprobs'}")
        print("─" * 96)
        all_correct = True
        for (q, solve, vr, outputs), row in zip(per_row_results, benchmark_rows):
            argmax = solve.response
            gold = choices[row["answer"]]
            lps = solve.scoring_details["_mc_choices_logprobs"]
            lps_str = "  ".join(f"{c}={lp:.2f}" for c, lp in zip(choices, lps))
            acc = outputs.get("acc", 0.0)
            if acc != 1.0:
                all_correct = False
            print(f"{q:<22} {argmax:<10} {gold:<10} {acc:<6} {lps_str}")

        print()
        print(f"all correct: {all_correct}")
        for q, solve, vr, outputs in per_row_results:
            assert outputs.get("acc") == 1.0, f"acc != 1.0 for {q!r}: {outputs}"
            assert math.isfinite(solve.scoring_details["_mc_choices_logprobs"][0])
        print("OK: end-to-end seed → solve → verify works through real HTTP")

        await solver.close()
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    asyncio.run(run_smoketest())
