# Example Configurations

End-to-end example configs for NeMo Evaluator. Each file demonstrates a distinct "flavor" — a unique combination of benchmark type, solver, verification method, and execution backend.

## Prerequisites

All configs expect `NVIDIA_API_KEY` (or `OPENAI_API_KEY`) set as an environment variable. SLURM configs require cluster access. ECS configs require AWS credentials.

## Config Index

| # | File | Benchmark | Solver | Backend | Key Feature |
|---|------|-----------|--------|---------|-------------|
| 01 | `01_gsm8k_chat.yaml` | GSM8K | simple | Local + API | Minimal text QA with numeric extraction |
| 02 | `02_mmlu_completions.yaml` | MMLU | simple (completions) | Local + API | Base model eval via `/completions` |
| 03 | `03_mmlu_lmeval.yaml` | lm-eval://mmlu | simple | Local + API | lm-evaluation-harness integration |
| 04 | `04_mmlu_pro_skills.yaml` | skills://mmlu-pro | simple | Local + API | NeMo Skills integration |
| 05 | `05_simpleqa_judge.yaml` | SimpleQA | simple | Local + API | Separate solver + judge models |
| 06 | `06_humaneval_docker.yaml` | HumanEval | simple | Docker sandbox | Sandboxed code execution |
| 07 | `07_swebench_harbor.yaml` | SWE-bench Verified | harbor | ECS Fargate | Agentic SWE with OpenHands on ECS |
| 07a | `07a_swebench_harbor_docker.yaml` | SWE-bench Verified | harbor | Docker | Local Docker variant of 07 |
| 08 | `08_terminalbench_harbor.yaml` | Terminal-Bench 2.0 | harbor | ECS Fargate | Agentic terminal tasks with Terminus-2 on ECS |
| 08a | `08a_terminalbench_harbor_docker.yaml` | Terminal-Bench 2.0 | harbor | Docker | Local Docker variant of 08 |
| 09 | `09_gym_runtime.yaml` | SWE-bench Multilingual | gym_delegation | Gym + Docker | Gym as agent runtime, nel verifies |
| 10 | `10_gym_full.yaml` | SWE-bench Multilingual | gym_delegation | Gym only | Gym runs agent + verification |
| 11 | `11_gym_env_harbor.yaml` | gym://localhost:8000 | harbor | Gym env + Docker | Gym seeds/verifies, nel runs agent |
| 12 | `12_pinchbench_openclaw.yaml` | PinchBench | openclaw | ECS Fargate | Agentic tasks + LLM-as-judge on ECS |
| 13 | `13_vlmevalkit_mmbench.yaml` | vlmevalkit://MMBench_DEV_EN | simple (VLM) | Local + API | VLMEvalKit VLM benchmark |
| 14 | `14_container_nemo_skills.yaml` | container://...#gsm8k | container | Docker | Legacy container harness |
| 15 | `15_slurm_gsm8k_vllm.yaml` | GSM8K | simple | SLURM + vLLM | Auto-deployed vLLM on cluster |
| 15a | `15a_slurm_gsm8k_vllm_sharded.yaml` | GSM8K | simple | SLURM + vLLM | Sharded eval via SLURM array jobs |
| 16 | `16_slurm_swebench_harbor.yaml` | SWE-bench Verified | harbor | SLURM + vLLM | Distributed agentic SWE on SLURM |
| 16a | `16a_slurm_swebench_harbor_apptainer.yaml` | SWE-bench Verified | harbor | SLURM + Apptainer | Apptainer variant of 16 |
| 17 | `17_suite_release.yaml` | GSM8K + MMLU + HumanEval + SimpleQA | simple | Mixed | Multi-benchmark release suite |
| 18 | `18_humaneval_ecs.yaml` | HumanEval | simple | ECS Fargate | Sandboxed code execution on AWS ECS |
| 19 | `19_gym_runtime_ecs.yaml` | SWE-bench Multilingual | gym_delegation | Gym + ECS | Gym agent runtime, nel verifies on ECS |
| 20 | `20_slurm_gym_runtime.yaml` | SWE-bench Multilingual | gym_delegation | SLURM + vLLM | Gym agent runtime, nel verifies on SLURM |
| 20a | `20a_slurm_gym_runtime_apptainer.yaml` | SWE-bench Multilingual | gym_delegation | SLURM + Apptainer | Apptainer variant of 20 |
| 21 | `21_tool_calling_gym.yaml` | SWE-bench Multilingual | tool_calling | Gym HTTP tools | NEL-driven ReAct loop with Gym Resource Server |
| 22 | `22_tool_calling_sandbox.yaml` | SWE-bench Verified | tool_calling | Docker sandbox | NEL as the coding agent (bash, file tools) |
| 23 | `23_tool_calling_combined.yaml` | Research bench | tool_calling | Gym + Docker | Combined HTTP + sandbox tools |

## Scenarios Covered

### By Solver Type
- **simple** (01, 02, 03, 04, 05, 06, 13, 15, 17): Standard chat/completions/VLM
- **harbor** (07, 07a, 08, 08a, 11, 16): Agentic evaluation via Harbor agents
- **tool_calling** (21, 22, 23): NEL-driven ReAct loop — full observability over model calls and tool dispatch
- **openclaw** (12): Agentic evaluation via OpenClaw
- **gym_delegation** (09, 10, 19, 20): Delegated to nemo-gym server
- **container** (14): Legacy container harness

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
- **Mode D** — NEL-driven tool calling (21, 23): ReActSolver drives the model loop; Gym Resource Server provides HTTP tools — full per-turn observability

## Running

```bash
# Single config
nel eval run examples/configs/01_gsm8k_chat.yaml

# Override service model at runtime
nel eval run examples/configs/01_gsm8k_chat.yaml -O services.nemotron.model=my-model

# Resume a partially completed suite
nel eval run examples/configs/17_suite_release.yaml --resume
```
