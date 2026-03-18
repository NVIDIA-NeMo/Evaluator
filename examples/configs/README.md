# Example Configurations

End-to-end example configs for NeMo Evaluator. Each file demonstrates a distinct "flavor" — a unique combination of benchmark type, solver, verification method, and execution backend.

## Prerequisites

All configs expect `NVIDIA_API_KEY` (or `OPENAI_API_KEY`) set as an environment variable. SLURM configs require cluster access. Docker sandbox configs require a running Docker daemon.

## Config Index

| # | File | Benchmark | Solver | Backend | Key Feature |
|---|------|-----------|--------|---------|-------------|
| 01 | `01_gsm8k_chat.yaml` | GSM8K | ChatSolver | Local + API | Minimal text QA with numeric extraction |
| 02 | `02_mmlu_completions.yaml` | MMLU | CompletionSolver | Local + API | Base model eval via `/completions` |
| 03 | `03_mmlu_lmeval.yaml` | lm-eval://mmlu | ChatSolver | Local + API | lm-evaluation-harness integration |
| 04 | `04_mmlu_pro_skills.yaml` | skills://mmlu-pro | ChatSolver | Local + API | NeMo Skills integration |
| 05 | `05_simpleqa_judge.yaml` | SimpleQA | ChatSolver | Local + API | LLM-as-judge scoring |
| 06 | `06_humaneval_docker.yaml` | HumanEval | ChatSolver | Docker sandbox | Sandboxed code execution |
| 07 | `07_swebench_harbor.yaml` | SWE-bench Verified | HarborSolver | Docker sandbox | Agentic SWE with OpenHands |
| 08 | `08_gym_runtime.yaml` | SWE-bench Multilingual | GymSolver | Gym + Docker | Gym as agent runtime, nel verifies |
| 09 | `09_gym_full.yaml` | SWE-bench Multilingual | GymSolver | Gym only | Gym runs agent + verification |
| 10 | `10_gym_env_harbor.yaml` | gym://localhost:8000 | HarborSolver | Gym env + Docker | Gym seeds/verifies, nel runs agent |
| 11 | `11_pinchbench_harbor.yaml` | PinchBench | HarborSolver | Docker sandbox | Agentic tasks + LLM-as-judge |
| 12 | `12_vlmevalkit_mmbench.yaml` | vlmevalkit://MMBench_DEV_EN | VLMSolver | Local + API | VLMEvalKit VLM benchmark |
| 13 | `13_container_nemo_skills.yaml` | container://...#gsm8k | (container) | Docker | Legacy container harness |
| 14 | `14_slurm_gsm8k_vllm.yaml` | GSM8K | ChatSolver | SLURM + vLLM | Auto-deployed vLLM on cluster |
| 15 | `15_slurm_swebench_harbor.yaml` | SWE-bench Verified | HarborSolver | SLURM + vLLM | Distributed agentic SWE on SLURM |
| 16 | `16_suite_release.yaml` | GSM8K + MMLU + HumanEval + SimpleQA | ChatSolver | Mixed | Multi-benchmark release suite |

## Scenarios Covered

### By Solver Type
- **ChatSolver** (01, 03, 04, 05, 06, 14, 16): Standard chat completions
- **CompletionSolver** (02): Raw text completions for base models
- **VLMSolver** (12): Vision-language model with images
- **HarborSolver** (07, 10, 11, 15): Agentic evaluation via Harbor/OpenHands
- **GymSolver** (08, 09): Delegated to nemo-gym server

### By Verification Method
- **Regex/numeric extraction** (01, 02, 03, 04, 14): Automated scoring
- **LLM-as-judge** (05, 11, 16): Model judges model
- **Docker sandbox** (06, 16): Sandboxed code execution
- **SWE-bench two-container** (07, 08, 15): Patch apply + test
- **Gym-trusted reward** (09): Gym's own verification
- **VLMEvalKit MCQ/VQA** (12): Option matching and VQA scoring
- **Container-owned** (13): Legacy harness handles everything

### By Execution Backend
- **Local + external API** (01–05, 12): NVIDIA API or any OpenAI-compatible endpoint
- **Local + Docker sandbox** (06, 07, 08, 10, 11, 13): Docker for code/agent execution
- **SLURM + auto-deployed vLLM** (14, 15): Full cluster deployment

### By nemo-gym Mode
- **Mode A** — Gym as agent runtime (08): GymSolver delegates agent execution; nel verifies
- **Mode B** — Gym as full environment (09): GymSolver trusts gym reward end-to-end
- **Mode C** — Gym as environment, nel as agent runtime (10): GymEnvironment for seed/verify; HarborSolver runs the agent

## Running

```bash
# Single config
nel eval run examples/configs/01_gsm8k_chat.yaml

# Override model at runtime
nel eval run examples/configs/01_gsm8k_chat.yaml -O model.id=my-model

# Resume a partially completed suite
nel eval run examples/configs/16_suite_release.yaml --resume
```
