# Example Configurations

End-to-end example configs for NeMo Evaluator. Each file demonstrates a distinct "flavor" — a unique combination of benchmark type, solver, verification method, and execution backend.

## Prerequisites

All configs expect `NVIDIA_API_KEY` (or `OPENAI_API_KEY`) set as an environment variable. SLURM configs require cluster access. ECS configs require AWS credentials.

## Config Index

| # | File | Benchmark | Solver | Backend | Key Feature |
|---|------|-----------|--------|---------|-------------|
| 01 | `01_gsm8k_chat.yaml` | GSM8K | ChatSolver | Local + API | Minimal text QA with numeric extraction |
| 02 | `02_mmlu_completions.yaml` | MMLU | CompletionSolver | Local + API | Base model eval via `/completions` |
| 03 | `03_mmlu_lmeval.yaml` | lm-eval://mmlu | ChatSolver | Local + API | lm-evaluation-harness integration |
| 04 | `04_mmlu_pro_skills.yaml` | skills://mmlu-pro | ChatSolver | Local + API | NeMo Skills integration |
| 05 | `05_simpleqa_judge.yaml` | SimpleQA | ChatSolver | Local + API | Separate solver + judge models |
| 06 | `06_humaneval_docker.yaml` | HumanEval | ChatSolver | Docker sandbox | Sandboxed code execution |
| 07 | `07_swebench_harbor.yaml` | SWE-bench Verified | HarborSolver | ECS Fargate | Agentic SWE with OpenHands on ECS |
| 07a | `07a_swebench_harbor_docker.yaml` | SWE-bench Verified | HarborSolver | Docker | Local Docker variant of 07 |
| 08 | `08_terminalbench_harbor.yaml` | Terminal-Bench 2.0 | HarborSolver | ECS Fargate | Agentic terminal tasks with Terminus-2 on ECS |
| 08a | `08a_terminalbench_harbor_docker.yaml` | Terminal-Bench 2.0 | HarborSolver | Docker | Local Docker variant of 08 |
| 09 | `09_gym_runtime.yaml` | SWE-bench Multilingual | GymSolver | Gym + Docker | Gym as agent runtime, nel verifies |
| 10 | `10_gym_full.yaml` | SWE-bench Multilingual | GymSolver | Gym only | Gym runs agent + verification |
| 11 | `11_gym_env_harbor.yaml` | gym://localhost:8000 | HarborSolver | Gym env + Docker | Gym seeds/verifies, nel runs agent |
| 12 | `12_pinchbench_openclaw.yaml` | PinchBench | OpenClawSolver | ECS Fargate | Agentic tasks + LLM-as-judge on ECS |
| 13 | `13_vlmevalkit_mmbench.yaml` | vlmevalkit://MMBench_DEV_EN | VLMSolver | Local + API | VLMEvalKit VLM benchmark |
| 14 | `14_container_nemo_skills.yaml` | container://...#gsm8k | (container) | Docker | Legacy container harness |
| 15 | `15_slurm_gsm8k_vllm.yaml` | GSM8K | ChatSolver | SLURM + vLLM | Auto-deployed vLLM on cluster |
| 15a | `15a_slurm_gsm8k_vllm_sharded.yaml` | GSM8K | ChatSolver | SLURM + vLLM | Sharded eval via SLURM array jobs |
| 16 | `16_slurm_swebench_harbor.yaml` | SWE-bench Verified | HarborSolver | SLURM + vLLM | Distributed agentic SWE on SLURM |
| 17 | `17_suite_release.yaml` | GSM8K + MMLU + HumanEval + SimpleQA | ChatSolver | Mixed | Multi-benchmark release suite |
| 18 | `18_humaneval_ecs.yaml` | HumanEval | ChatSolver | ECS Fargate | Sandboxed code execution on AWS ECS |
| 19 | `19_gym_runtime_ecs.yaml` | SWE-bench Multilingual | GymSolver | Gym + ECS | Gym agent runtime, nel verifies on ECS |
| 20 | `20_slurm_gym_runtime.yaml` | SWE-bench Multilingual | GymSolver | SLURM + vLLM | Gym agent runtime, nel verifies on SLURM |

## Scenarios Covered

### By Solver Type
- **ChatSolver** (01, 03, 04, 05, 06, 15, 17): Standard chat completions
- **CompletionSolver** (02): Raw text completions for base models
- **VLMSolver** (13): Vision-language model with images
- **HarborSolver** (07, 07a, 08, 08a, 11, 16): Agentic evaluation via Harbor agents
- **OpenClawSolver** (12): Agentic evaluation via OpenClaw
- **GymSolver** (09, 10, 19, 20): Delegated to nemo-gym server

### By Verification Method
- **Regex/numeric extraction** (01, 02, 03, 04, 15): Automated scoring
- **LLM-as-judge** (05, 12, 17): Model judges model
- **Docker sandbox** (06, 17): Sandboxed code execution
- **SWE-bench two-container** (07, 08, 09, 16, 19, 20): Patch apply + test
- **Gym-trusted reward** (10): Gym's own verification
- **VLMEvalKit MCQ/VQA** (13): Option matching and VQA scoring
- **Container-owned** (14): Legacy harness handles everything

### By Execution Backend
- **Local + external API** (01–05, 13): NVIDIA API or any OpenAI-compatible endpoint
- **Local + Docker sandbox** (06, 07a, 08a, 09, 11, 14): Docker for code/agent execution
- **AWS ECS Fargate** (07, 08, 12, 18, 19): Remote sandbox on ECS — no local Docker needed
- **SLURM + auto-deployed vLLM** (15, 16, 20): Full cluster deployment

### By nemo-gym Mode
- **Mode A** — Gym as agent runtime (09): GymSolver delegates agent execution; nel verifies
- **Mode B** — Gym as full environment (10): GymSolver trusts gym reward end-to-end
- **Mode C** — Gym as environment, nel as agent runtime (11): GymEnvironment for seed/verify; HarborSolver runs the agent

## Running

```bash
# Single config
nel eval run examples/configs/01_gsm8k_chat.yaml

# Override model at runtime
nel eval run examples/configs/01_gsm8k_chat.yaml -O model.id=my-model

# Resume a partially completed suite
nel eval run examples/configs/17_suite_release.yaml --resume
```
