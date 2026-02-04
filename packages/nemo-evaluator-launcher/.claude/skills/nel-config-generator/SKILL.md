---
name: nel-config-generator
description: Interactive config generator for NeMo Evaluator Launcher. Creates YAML configs for running model evaluations with support for SLURM and Local executors, all deployment types (vLLM, SGLang, NIM, TRT-LLM, None), and all public exporters (MLflow, W&B, GSheets, Local). Use when users want to create evaluation configs, set up model evaluations, run benchmarks, or generate nel configs.
---

# NeMo Evaluator Launcher Config Generator

Guide users through creating production-ready YAML configurations via an interactive 7-question workflow.

## Before Starting

**1. Get current task list with containers:**
```bash
nel ls tasks
```

If `nel` command is not found, ask user to install from PyPI:
```bash
pip install nemo-evaluator-launcher[all]
```

Use the task list output to populate task containers and warn if presets are outdated.

## Support

Direct users with issues to:
- **GitHub Issues:** https://github.com/NVIDIA-NeMo/Evaluator/issues
- **GitHub Discussions:** https://github.com/NVIDIA-NeMo/Evaluator/discussions

## Documentation Sources

For current configuration details:
- **Public docs:** Fetch from `https://docs.nvidia.com/nemo/evaluator/latest/llms.txt` for configuration details
- **Example configs:** Fetch from GitHub at `https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/refs/heads/main/packages/nemo-evaluator-launcher/examples/`
- **Task list:** Run `nel ls tasks` for current tasks with container images

### Key Example Files

Fetch examples from GitHub using the base URL: `https://raw.githubusercontent.com/NVIDIA-NeMo/Evaluator/refs/heads/main/packages/nemo-evaluator-launcher/examples/`

| Scenario | Example File |
|----------|--------------|
| Local basic | `local_basic.yaml` |
| Local with reasoning | `local_reasoning.yaml` |
| SLURM basic | `slurm_vllm_basic.yaml` |
| SLURM multi-node DP | `slurm_vllm_multinode_dp.yaml` |
| SLURM multi-node Ray | `slurm_vllm_multinode_ray_tp_pp.yaml` |
| Adapters | `local_advanced_adapters.yaml` |
| Auto export | `local_auto_export.yaml` |

---

## Workflow Overview

Guide the user through 7 questions using AskUserQuestion, then generate the config file.

### Question 1: Executor

**Question:** "Where will the evaluation run?"
- **SLURM** (HPC cluster - recommended for large models)
- **Local** (Docker on local machine)

**If SLURM selected, ask follow-up questions:**
1. "What is the SLURM cluster hostname?" (required)
2. "What is the SLURM account name?" (required)
3. "What is the output directory path?" (required - must be on shared filesystem)
4. "How many GPUs per node?" (default: 8)

### Question 2: Deployment Type

Fetch the appropriate example config from GitHub for detailed deployment examples (see Key Example Files table above).

**Question:** "How will the model be served?"
- **vLLM** (recommended for most models)
- **SGLang** (structured generation)
- **NIM** (NVIDIA Inference Microservices)
- **TensorRT-LLM** (optimized inference)
- **Generic** (bring-your-own server)
- **None** (external API endpoint already deployed)

**If None or Generic selected, follow-up:**
1. Ask for API URL (e.g., `https://api.openai.com/v1`)
2. Ask for model_id (e.g., `gpt-4o`)
3. Ask for api_key_name (environment variable name)

### Question 3: Model

**Question:** "What is the model path? (HuggingFace handle or checkpoint path)"

This is free-form text input. The user must provide either:
- **HuggingFace handle:** Format is `org/model-name` (contains `/` but does NOT start with `/`)
- **Checkpoint path:** Starts with `/` (absolute) or `./` (relative)

**Note:** Do not provide pre-populated model options.

**IMPORTANT - Setting deployment fields:**
- If HuggingFace handle: set `hf_model_handle: <handle>` and `checkpoint_path: null`
- If checkpoint path: set `checkpoint_path: <path>` and `hf_model_handle: null`
- **Both fields must always be present** - the unused one must be explicitly set to `null`

**After receiving the model path:**
1. **Auto-detect model information:** Use WebSearch to find the model card and determine:
   - Parameter count (e.g., 8B, 70B, 235B)
   - Architecture (dense vs MoE)
   - Reasoning capability (thinking tokens)
2. **Only ask for clarification** if unable to determine size with high confidence
3. **Auto-configure** based on detected size:

| Model Size | TP | PP | DP | Nodes | Parallelism |
|------------|----|----|----|----|-------------|
| Small (1-15B) | 1-2 | 1 | 1 | 1 | 64-128 |
| Medium (15-70B) | 4-8 | 1 | 1 | 1 | 128-256 |
| Large (70-200B) | 8 | 2 | 1 | 2 | 256-512 |
| MoE/Giant (200B+) | 4-8 | 1 | 4-8 | 4-8 | 512 |

**For checkpoint paths:** Ask for model size category and if model supports reasoning.

**Important:** If using local executor with checkpoint_path, add `mounts` under `execution`.

### Question 4: Adapter Config

Fetch `local_advanced_adapters.yaml` from GitHub for adapter configuration examples. Fetch docs from `https://docs.nvidia.com/nemo/evaluator/latest/llms.txt` for current adapter options.

**Question:** "Configure adapter settings?"
- Use defaults for model (recommended)
- Enable reasoning extraction (for thinking models)
- Disable reasoning processing
- Custom configuration

**If Custom or reasoning models, follow-up:** Ask about reasoning tokens, logging, caching.

### Question 5: Tasks (MULTI-SELECT)

**Question:** "Which benchmarks do you want to run? (select multiple)"
- Standard LLM Benchmarks (MMLU, IFEval, GSM8K, etc.)
- Code Evaluation (HumanEval, MBPP, LiveCodeBench)
- Math & Reasoning (AIME, GPQA, MATH-500)
- Safety & Security (Garak, Safety Harness)
- **Search for specific tasks** (keyword search)

**If "Search for specific tasks" selected:**
1. Ask user for search keywords
2. Run `nel ls tasks | grep -i <keyword>` to filter results
3. Present matching tasks with task name and harness name

### Question 6: Exporter

**Question:** "Where should results be exported?"
- **MLflow** (experiment tracking - requires tracking_uri)
- **Weights & Biases** (W&B visualization/tracking)
- **Google Sheets** (shared spreadsheets)
- **Local** (save to filesystem)

**If MLflow selected:** Ask for tracking_uri.
**If W&B selected:** Ask for project name and entity (optional).
**If Google Sheets selected:** Ask for spreadsheet_id.

### Question 7: Parameters

Fetch defaults from model card via web search, then present for review.

**Question:** "Review and adjust model parameters?"
- Show detected defaults: "Based on [model] card: temperature=X, top_p=Y"
- Options: Use these defaults (recommended) / Customize parameters

**If Customize, allow adjusting:**
- temperature (0.6 for reasoning, 0.0 for deterministic)
- top_p (default: 0.95)
- top_k (if specified in model card)
- max_new_tokens (varies by task: 8K-256K)
- parallelism (based on model size: 64-512)
- request_timeout (default: 100000)

---

## Config Generation

When generating configs:
1. Fetch the appropriate example file from GitHub based on executor and deployment type (see Key Example Files table)
2. Fetch specific details from docs at `https://docs.nvidia.com/nemo/evaluator/latest/llms.txt` if needed
3. Adapt the example to user's requirements

**Key points:**
- SLURM uses `execution: slurm/default`, Local uses `execution: local`
- Always specify both `checkpoint_path` and `hf_model_handle` (set unused to `null`)
- For multi-node deployments, fetch `slurm_vllm_multinode_dp.yaml` or `slurm_vllm_multinode_ray_tp_pp.yaml` from GitHub

---

## Required Environment Variables

| Variable | When Required |
|----------|--------------|
| HF_TOKEN | Gated HuggingFace models |
| NGC_API_KEY | NVIDIA API endpoints |
| OPENAI_API_KEY | OpenAI API endpoints |
| JUDGE_API_KEY | Tasks requiring LLM-as-judge |
| WANDB_API_KEY | W&B exporter |

---

## Output File Generation

1. Ask user for output filename (default: `{model_short_name}_{executor}.yaml`)
2. Write the config to the current working directory or user-specified path
3. Show summary: executor, model, parallelism, tasks, exporter
4. List required environment variables
5. Suggest validation: `nel run --config <filename> --dry-run`
