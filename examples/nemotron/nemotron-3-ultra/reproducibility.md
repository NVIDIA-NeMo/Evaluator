# NVIDIA Nemotron 3 Ultra 550B A55B — Reproducing Model Card Evaluation Results

This tutorial demonstrates how to reproduce the evaluation results for the NVIDIA Nemotron 3 Ultra 550B A55B model using NeMo Evaluator.

This notebook uses two distinct **NeMo Evaluator generations**: **v0.2** (`nemo-evaluator-launcher` 0.2.6) and **v0.3** (`nemo-evaluator` 0.3.x). Benchmarks are run in either one generation or the other, and that is made clear in the table further below.

The instruct suite is run on v0.2; the agentic benchmarks differ — v0.2 = Gym (LiveCodeBench, Tau2, GDPVal), v0.3 = native SWE-bench + Terminal-bench.

Each recipe ships with its own README — setup, API keys, running, and per-benchmark notes live there:

- **v0.2:** [`./v0.2/README.md`](./v0.2/README.md) — [common setup](./v0.2/README.md#common-setup) · [running](./v0.2/README.md#running) · [benchmarks](./v0.2/README.md#benchmarks)
- **v0.3:** [`./v0.3/README.md`](./v0.3/README.md) — [common setup](./v0.3/README.md#common-setup) · [running](./v0.3/README.md#running) · [benchmarks](./v0.3/README.md#benchmarks)

## Overview

The **Recipe** column links to each benchmark's recipe — the v0.2 README sections for the instruct and Gym agentic benchmarks, or the v0.3 config for the native agentic benchmarks.

| Benchmark | Category | Description | Recipe |
|---|---|---|---|
| GPQA (Diamond) | Science | Graduate-level science questions | Instruct ([v0.2](./v0.2/README.md#gpqa-diamond)) |
| HLE | Humanity's Last Exam | Expert-level questions across domains | Instruct ([v0.2](./v0.2/README.md#hle)) |
| MMLU-Pro | Knowledge | Multi-task language understanding (10-choice) | Instruct ([v0.2](./v0.2/README.md#mmlu-pro)) |
| MMLU-Pro X | Knowledge | MMLU-Pro extended multilingual variant | Instruct ([v0.2](./v0.2/README.md#mmlu-pro-x)) |
| SciCode | Scientific Coding | Scientific programming challenges | Instruct ([v0.2](./v0.2/README.md#scicode)) |
| CritPt | Physics | Critical Point evaluation | Instruct ([v0.2](./v0.2/README.md#critpt)) |
| IFBench | Instruction Following | Instruction following benchmark | Instruct ([v0.2](./v0.2/README.md#ifbench)) |
| AA-Omniscience | Knowledge / hallucination | AA Omniscience benchmark (Gemini judge) | Instruct ([v0.2](./v0.2/README.md#aa-omniscience)) |
| IMO-AnswerBench | Mathematics | International Math Olympiad answer benchmark | Instruct ([v0.2](./v0.2/README.md#imo-answerbench)) |
| AA-LCR | Long-context | Artificial Analysis long-context benchmark | Instruct ([v0.2](./v0.2/README.md#aa-lcr)) |
| Multi-Challenge | Multi-turn | Multi-turn instruction-following benchmark | Instruct ([v0.2](./v0.2/README.md#multi-challenge)) |
| WMT24++ | Translation | WMT24 translation, scored with XCOMET-XXL | Instruct ([v0.2](./v0.2/README.md#wmt24)) |
| LiveCodeBench v6 | Coding | Real-world coding problems | Agentic — [v0.2 (Gym)](./v0.2/README.md#livecodebench-v6-cascade) |
| Tau2 | Tool Use | Tool use across telecom / airline / retail | Agentic — [v0.2 (Gym)](./v0.2/README.md#tau2-taubench-v3) |
| GDPVal | Office-deliverable agent | Office/PDF deliverables, judged pairwise/rubric | Agentic — [v0.2 (Gym)](./v0.2/README.md#gdpval-stirrup-agent) |
| SWE-bench (Verified / Multilingual / Pro) | Agentic coding | OpenHands agent fixes a repo; graded by the task's tests | Agentic — [v0.3-native (harbor)](./v0.3/README.md#swe-bench-verified--multilingual--pro) |
| Terminal-bench (Hard / 2.0) | Agentic terminal | Terminus-2 agent drives a shell; graded by `test.sh` | Agentic — [v0.3-native (harbor)](./v0.3/README.md#terminal-bench-hard--20) |
| Base suite | Pretraining | MMLU, GSM8K, HumanEval, RULER, … (21 tasks) | [Base — v0.2 (local vLLM)](./v0.2/README.md#base-suite) |

The following benchmarks are not onboarded yet in our open source tools and for these we used either their official open source implementation or otherwise an internal scaffolding that we plan to open source in the future: BrowseComp, Tau Bench 3, ProfBench, PinchBench, Vals.ai Financial Agent, LongBench v2.

## Endpoint

- **Hosted:** `https://integrate.api.nvidia.com/v1/chat/completions`, model `nvidia/nemotron-3-ultra-550b-a55b`, key `NGC_API_KEY` (v0.2) / `NVIDIA_API_KEY` (v0.3).
- **Self-hosted vLLM:** serve `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-{BF16,FP8,NVFP4}`; full serve commands (parsers, TP, env vars) are in the header of `./v0.2/local_nemotron-3-ultra-550b-a55b.yaml`.

Default instruct params (same in v0.2 and v0.3): `temperature 1.0`, `top_p 0.95`, `max_new_tokens 262144`, `parallelism 1`, `request_timeout 3600`, `max_retries 10`, thinking on.

## License

Apache 2.0 — see the repository `LICENSE`.
