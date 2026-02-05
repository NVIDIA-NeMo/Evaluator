# Plan: Create NEL Config Generator Asset Files

## Overview
Create configuration snippet files in `.claude/skills/nel-config-generator/assets/` that will be used as building blocks for generating full NEL configs based on user choices in Step 2 of the SKILL.md workflow.

## Directory Structure (already exists)
```
assets/
├── deployment/
├── execution/
├── export/
└── evaluation/
    ├── base/       # completions endpoint tasks only
    ├── chat/       # chat endpoint tasks
    └── reasoning/  # chat endpoint tasks + reasoning config
```

## Key Insight: Task Endpoint Types
Tasks are categorized by their endpoint type:
- **completions** - For base models (non-chat)
- **chat** - For chat/instruct models and reasoning models

Benchmarks must be placed in the appropriate model type directory based on endpoint compatibility.

## Files to Create

### 1. Execution Options (Step 2, Q1)

#### `execution/local.yaml`
```yaml
defaults:
  - execution: local

execution:
  output_dir: nel-results
```

#### `execution/slurm.yaml`
```yaml
defaults:
  - execution: slurm/default

execution:
  hostname: ??? # SLURM headnode (login) hostname (required)
  username: ${oc.env:USER} # Cluster username; defaults to $USER
  account: ??? # SLURM account allocation (required)
  output_dir: ??? # ABSOLUTE path accessible to SLURM compute nodes (required)
  mounts:
    mount_home: false # Recommended: don't mount home directory on cluster
```

---

### 2. Deployment Options (Step 2, Q2)

#### `deployment/none.yaml` (External API)
```yaml
defaults:
  - deployment: none

target:
  api_endpoint:
    model_id: ??? # Model identifier (e.g., meta/llama-3.2-3b-instruct)
    url: ??? # API endpoint URL (e.g., https://integrate.api.nvidia.com/v1/chat/completions)
    api_key_name: ??? # Environment variable name containing API key (e.g., NGC_API_KEY)
```

#### `deployment/vllm.yaml`
```yaml
defaults:
  - deployment: vllm

execution:
  env_vars:
    deployment:
      HF_TOKEN: ${oc.env:HF_TOKEN} # Required for gated HuggingFace models

deployment:
  checkpoint_path: null # Set to path if using local checkpoint
  hf_model_handle: ??? # HuggingFace model handle (e.g., meta-llama/Llama-3.1-8B)
  served_model_name: ??? # Model name for API (e.g., meta-llama/Llama-3.1-8B)
  tensor_parallel_size: 1
  data_parallel_size: 1
  extra_args: "--max-model-len 32768"
```

#### `deployment/sglang.yaml`
```yaml
defaults:
  - deployment: sglang

execution:
  env_vars:
    deployment:
      HF_TOKEN: ${oc.env:HF_TOKEN} # Required for gated HuggingFace models

deployment:
  checkpoint_path: null # Set to path if using local checkpoint
  hf_model_handle: ??? # HuggingFace model handle
  served_model_name: ??? # Model name for API
  tensor_parallel_size: 1
  data_parallel_size: 1
  extra_args: "--context-length 32768"
```

#### `deployment/nim.yaml`
Includes execution adjustments for NIM-specific requirements:
```yaml
defaults:
  - deployment: nim

execution:
  env_vars:
    deployment:
      NGC_API_KEY: ${oc.env:NGC_API_KEY} # Required for NIM container authentication
  mounts:
    deployment:
      /path/to/nim/cache: /opt/nim/.cache # Replace with absolute path to NIM cache directory

deployment:
  image: ??? # NIM image (e.g., nvcr.io/nim/meta/llama-3.2-1b-instruct:latest)
  served_model_name: ??? # Model name (e.g., meta/llama-3.2-1b-instruct)
```

#### `deployment/trtllm.yaml`
```yaml
defaults:
  - deployment: trtllm

deployment:
  checkpoint_path: ??? # Path to TRT-LLM engine/checkpoint (required)
  served_model_name: ??? # Model name for API
  tensor_parallel_size: 8
  pipeline_parallel_size: 1
  extra_args: ""
```

---

### 3. Export Options (Step 2, Q3)

#### `export/none.yaml`
Empty file or minimal content (no auto-export):
```yaml
# No auto-export configured
# Results will be saved locally to execution.output_dir
```

#### `export/mlflow.yaml`
```yaml
execution:
  auto_export:
    destinations: ["mlflow"]

export:
  mlflow:
    tracking_uri: ??? # MLflow server URL (http://hostname:port) (required)
    experiment_name: "nemo-evaluator-launcher" # Optional
    description: "" # Optional
    log_logs: true
    log_config_params: true
    tags: {} # Optional key-value pairs
```

#### `export/wandb.yaml`
```yaml
execution:
  auto_export:
    destinations: ["wandb"]

export:
  wandb:
    project: ??? # W&B project name (required)
    entity: ??? # W&B entity/team name (optional, defaults to user)
    log_logs: true
    tags: {} # Optional key-value pairs
```

---

### 4. Evaluation / Model Type + Benchmarks (Step 2, Q4 & Q5)

Benchmarks are organized under model type directories based on endpoint compatibility:
- **Base models** use `completions` endpoint tasks
- **Chat models** use `chat` endpoint tasks
- **Reasoning models** use `chat` endpoint tasks with special reasoning config

---

#### BASE MODEL CONFIGS (`evaluation/base/`)

Base models are typically evaluated using **log-probabilities** rather than text generation.
This requires:
- A **completions endpoint** (not chat) that supports `logprobs` and `echo` parameters
- **Tokenizer configuration** for client-side tokenization

See: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/logprobs.html

##### `evaluation/base/default.yaml` - Global params for base models
```yaml
# Base model evaluation configuration (completions endpoint)
# Uses log-probabilities for multiple-choice tasks
# See: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/logprobs.html
evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        parallelism: 64
        extra:
          # Tokenizer required for log-probability tasks
          tokenizer: ??? # HuggingFace model handle or path to tokenizer (must match evaluated model)
          tokenizer_backend: huggingface # or "tiktoken"
  env_vars:
    HF_TOKEN: HF_TOKEN # Required to access gated tokenizers
```

##### `evaluation/base/standard.yaml` - Standard LLM Benchmarks (completions/logprobs)
```yaml
# Standard LLM Benchmarks for base models (completions endpoint, log-probability based)
# These tasks use log-probabilities to assess model confidence on answer choices
evaluation:
  tasks:
    - name: mmlu # Log-prob based multiple choice
    - name: gpqa # Log-prob based (completions version)
    - name: arc_challenge # Log-prob based
    - name: hellaswag # Log-prob based
    - name: winogrande # Log-prob based
    - name: piqa # Log-prob based
    - name: openbookqa # Log-prob based
    - name: commonsense_qa # Log-prob based
```

##### `evaluation/base/code.yaml` - Code Evaluation (completions)
```yaml
# Code Evaluation for base models (completions endpoint)
# Note: Code tasks use text generation, not log-probabilities
evaluation:
  tasks:
    - name: humaneval # completions version - text generation
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.2
            top_p: 0.95
            max_new_tokens: 2048
            extra:
              n_samples: 5 # Sample multiple predictions for pass@k
    - name: mbpp # completions version - text generation
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.2
            top_p: 0.95
            max_new_tokens: 2048
            extra:
              n_samples: 5
```

##### `evaluation/base/multilingual.yaml` - Multilingual Benchmarks (completions/logprobs)
```yaml
# Multilingual Benchmarks for base models (completions endpoint, log-probability based)
evaluation:
  tasks:
    - name: global_mmlu # Global-MMLU-Lite (10 languages subset) - log-prob based
    - name: global_mmlu_full # Global-MMLU full (42 languages) - log-prob based
    - name: arc_multilingual # Log-prob based
    - name: hellaswag_multilingual # Log-prob based
```

---

#### CHAT MODEL CONFIGS (`evaluation/chat/`)

##### `evaluation/chat/default.yaml` - Global params for chat models
```yaml
# Chat model evaluation configuration (chat endpoint)
evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        parallelism: 64
        temperature: 0.0 # Deterministic for reproducibility
        max_new_tokens: 2048
```

##### `evaluation/chat/standard.yaml` - Standard LLM Benchmarks (chat)
```yaml
# Standard LLM Benchmarks for chat models (chat endpoint)
evaluation:
  tasks:
    - name: ifeval
    - name: gpqa_diamond # chat version with CoT
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND # Request access: https://huggingface.co/datasets/Idavidrein/gpqa
    - name: gsm8k_cot_instruct # chat version with CoT
    - name: mmlu_pro # chat version
```

##### `evaluation/chat/code.yaml` - Code Evaluation (chat)
```yaml
# Code Evaluation for chat models (chat endpoint)
evaluation:
  tasks:
    - name: humaneval_instruct # chat version
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.2
            top_p: 0.95
            max_new_tokens: 2048
            extra:
              n_samples: 5
    - name: mbpp_plus # chat version (EvalPlus)
      nemo_evaluator_config:
        config:
          params:
            temperature: 0.2
            top_p: 0.95
            max_new_tokens: 2048
            extra:
              n_samples: 5
    - name: livecodebench_0824_0225
```

##### `evaluation/chat/math_reasoning.yaml` - Math & Reasoning (chat)
```yaml
# Math & Reasoning for chat models (chat endpoint)
# Note: For reasoning models, use evaluation/reasoning/ configs instead
evaluation:
  tasks:
    - name: gpqa_diamond
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
    - name: math_test_500
    - name: aime_2024_nemo
    - name: aime_2025_nemo
```

##### `evaluation/chat/safety.yaml` - Safety & Security (chat)
```yaml
# Safety & Security for chat models (chat endpoint)
evaluation:
  tasks:
    - name: garak
      # Uncomment for quick testing:
      # nemo_evaluator_config:
      #   config:
      #     params:
      #       extra:
      #         probes: "test.Test"
    - name: aegis_v2
      nemo_evaluator_config:
        config:
          params:
            extra:
              judge:
                model_id: "llama-3.1-nemoguard-8b-content-safety"
                url: ??? # Set to your judge endpoint URL (e.g., https://my-judge-endpoint.com/v1)
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_AEGIS_V2 # Access: https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0
        JUDGE_API_KEY: JUDGE_API_KEY # API key for judge endpoint
```

##### `evaluation/chat/multilingual.yaml` - Multilingual Benchmarks (chat)
```yaml
# Multilingual Benchmarks for chat models (chat endpoint)
evaluation:
  tasks:
    - name: mgsm_cot # Multilingual GSM8K with CoT (chat)
    - name: mmlu_prox # MMLU-Prox multilingual (6 languages)
    - name: mmath_en # Multilingual MATH - English
    - name: mmath_zh # Multilingual MATH - Chinese
    - name: mmath_ja # Multilingual MATH - Japanese
    - name: mmath_ko # Multilingual MATH - Korean
    - name: mmath_es # Multilingual MATH - Spanish
    - name: mmath_fr # Multilingual MATH - French
```

---

#### REASONING MODEL CONFIGS (`evaluation/reasoning/`)

Reasoning models require special configuration:
- Extended `max_new_tokens` for reasoning trace + final answer
- Sampling params (temperature=0.6, top_p=0.95) per model card
- `process_reasoning_traces: true` to strip thinking tokens
- System prompt to enable reasoning (e.g., "/think")

##### `evaluation/reasoning/default.yaml` - Global params for reasoning models
```yaml
# Reasoning model evaluation configuration (chat endpoint + reasoning)
# See docs: https://docs.nvidia.com/nemo/evaluator/latest/evaluation/run-evals/reasoning.html
evaluation:
  nemo_evaluator_config:
    config:
      params:
        request_timeout: 3600
        parallelism: 32 # Lower parallelism for longer generations
        temperature: 0.6 # Recommended for reasoning models
        top_p: 0.95
        max_new_tokens: 32768 # Extended for reasoning + final answer
    target:
      api_endpoint:
        adapter_config:
          process_reasoning_traces: true # Strip reasoning tokens and collect stats
          use_system_prompt: true
          custom_system_prompt: "/think" # Enable reasoning mode
```

##### `evaluation/reasoning/standard.yaml` - Standard LLM Benchmarks (reasoning)
```yaml
# Standard LLM Benchmarks for reasoning models
# Uses reasoning-enabled config from default.yaml
evaluation:
  tasks:
    - name: gpqa_diamond_nemo
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
    - name: mmlu_pro
      nemo_evaluator_config:
        config:
          params:
            max_new_tokens: 32768 # Extended for reasoning
    # IFEval should run with reasoning OFF for accuracy
    - name: ifeval
      nemo_evaluator_config:
        config:
          params:
            temperature: 0
            top_p: 1e-5
            max_new_tokens: 4096
        target:
          api_endpoint:
            adapter_config:
              use_system_prompt: true
              custom_system_prompt: "/no_think" # Disable reasoning for IFEval
```

##### `evaluation/reasoning/code.yaml` - Code Evaluation (reasoning)
```yaml
# Code Evaluation for reasoning models
evaluation:
  tasks:
    - name: mbppplus_nemo # NeMo version optimized for reasoning
    - name: livecodebench_0824_0225
      nemo_evaluator_config:
        config:
          params:
            max_new_tokens: 65536 # Extended for complex code reasoning
```

##### `evaluation/reasoning/math_reasoning.yaml` - Math & Reasoning (reasoning)
```yaml
# Math & Reasoning for reasoning models
# These benchmarks benefit most from extended thinking
evaluation:
  tasks:
    - name: gpqa_diamond_nemo
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
    - name: math_test_500_nemo
    - name: aime_2024_nemo
    - name: aime_2025_nemo
```

##### `evaluation/reasoning/safety.yaml` - Safety & Security (reasoning)
```yaml
# Safety & Security for reasoning models
# Consider running safety evals with reasoning OFF
evaluation:
  tasks:
    - name: garak
      nemo_evaluator_config:
        config:
          params:
            max_new_tokens: 4096 # Safety evals don't need extended reasoning
        target:
          api_endpoint:
            adapter_config:
              use_system_prompt: true
              custom_system_prompt: "/no_think" # Disable reasoning for safety
    - name: aegis_v2
      nemo_evaluator_config:
        config:
          params:
            max_new_tokens: 4096
            extra:
              judge:
                model_id: "llama-3.1-nemoguard-8b-content-safety"
                url: ??? # Set to your judge endpoint URL
        target:
          api_endpoint:
            adapter_config:
              use_system_prompt: true
              custom_system_prompt: "/no_think"
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_AEGIS_V2
        JUDGE_API_KEY: JUDGE_API_KEY
```

##### `evaluation/reasoning/multilingual.yaml` - Multilingual Benchmarks (reasoning)
```yaml
# Multilingual Benchmarks for reasoning models
# Math benchmarks benefit from reasoning; others run with reasoning off
evaluation:
  tasks:
    # Math benchmarks with reasoning ON
    - name: mmath_en
    - name: mmath_zh
    - name: mmath_ja
    - name: mmath_ko
    # MGSM with reasoning ON
    - name: mgsm_cot
    # MMLU-Prox with reasoning OFF (factual recall)
    - name: mmlu_prox
      nemo_evaluator_config:
        config:
          params:
            temperature: 0
            max_new_tokens: 4096
        target:
          api_endpoint:
            adapter_config:
              use_system_prompt: true
              custom_system_prompt: "/no_think"
```

---

### 6. Config Builder Script

#### `.claude/skills/nel-config-generator/scripts/build_config.py`

Python script that combines selected config excerpts into a single YAML file.

```python
#!/usr/bin/env python3
"""
Build a complete NEL config by combining config excerpts from assets.

Usage:
    python build_config.py \
        --execution local \
        --deployment vllm \
        --export mlflow \
        --model-type chat \
        --benchmarks standard code \
        --output my_config.yaml
"""

import argparse
from pathlib import Path
from typing import List, Optional
import yaml


ASSETS_DIR = Path(__file__).parent.parent / "assets"


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge two dictionaries, with override taking precedence."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        elif key in result and isinstance(result[key], list) and isinstance(value, list):
            # For lists (like tasks), extend rather than replace
            result[key] = result[key] + value
        else:
            result[key] = value
    return result


def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents."""
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


def build_config(
    execution: str,
    deployment: str,
    export: str,
    model_type: str,
    benchmarks: List[str],
    output: Optional[Path] = None,
) -> dict:
    """
    Build a complete NEL config by combining excerpts.

    Args:
        execution: Execution type (local, slurm)
        deployment: Deployment type (none, vllm, sglang, nim, trtllm)
        export: Export type (none, mlflow, wandb)
        model_type: Model type (base, chat, reasoning)
        benchmarks: List of benchmark types (standard, code, math_reasoning, safety)
        output: Optional output file path

    Returns:
        Combined config dictionary
    """
    config = {}

    # 1. Load execution config
    execution_path = ASSETS_DIR / "execution" / f"{execution}.yaml"
    config = deep_merge(config, load_yaml(execution_path))

    # 2. Load deployment config
    deployment_path = ASSETS_DIR / "deployment" / f"{deployment}.yaml"
    config = deep_merge(config, load_yaml(deployment_path))

    # 3. Load export config
    export_path = ASSETS_DIR / "export" / f"{export}.yaml"
    config = deep_merge(config, load_yaml(export_path))

    # 4. Load model type default config
    model_default_path = ASSETS_DIR / "evaluation" / model_type / "default.yaml"
    config = deep_merge(config, load_yaml(model_default_path))

    # 5. Load benchmark configs
    for benchmark in benchmarks:
        benchmark_path = ASSETS_DIR / "evaluation" / model_type / f"{benchmark}.yaml"
        if benchmark_path.exists():
            config = deep_merge(config, load_yaml(benchmark_path))
        else:
            print(f"Warning: Benchmark config not found: {benchmark_path}")

    # 6. Write output if specified
    if output:
        with open(output, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        print(f"Config written to: {output}")

    return config


def main():
    parser = argparse.ArgumentParser(
        description="Build NEL config from excerpts"
    )
    parser.add_argument(
        "--execution", "-e",
        choices=["local", "slurm"],
        required=True,
        help="Execution type"
    )
    parser.add_argument(
        "--deployment", "-d",
        choices=["none", "vllm", "sglang", "nim", "trtllm"],
        required=True,
        help="Deployment type"
    )
    parser.add_argument(
        "--export", "-x",
        choices=["none", "mlflow", "wandb"],
        default="none",
        help="Export type (default: none)"
    )
    parser.add_argument(
        "--model-type", "-m",
        choices=["base", "chat", "reasoning"],
        required=True,
        help="Model type"
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        choices=["standard", "code", "math_reasoning", "safety", "multilingual"],
        required=True,
        help="Benchmark types to include"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output file path"
    )
    parser.add_argument(
        "--validate", "-v",
        action="store_true",
        help="Validate the generated config using verify_config.py"
    )

    args = parser.parse_args()

    config = build_config(
        execution=args.execution,
        deployment=args.deployment,
        export=args.export,
        model_type=args.model_type,
        benchmarks=args.benchmarks,
        output=args.output,
    )

    if not args.output:
        # Print to stdout if no output file specified
        print(yaml.dump(config, default_flow_style=False, sort_keys=False))

    # Optionally validate the generated config
    if args.validate and args.output:
        from verify_config import resolve_config, validate_config
        try:
            cfg = resolve_config(str(args.output))
            valid, errors, warnings = validate_config(cfg)
            if valid:
                print("✓ Configuration is valid!")
                for warn in warnings:
                    print(f"  ⚠ Warning: {warn}")
            else:
                print("✗ Configuration has errors:")
                for err in errors:
                    print(f"  - {err}")
        except Exception as e:
            print(f"✗ Validation failed: {e}")


if __name__ == "__main__":
    main()
```

**Features:**
- Deep merges configs (nested dicts merged, task lists extended)
- Validates choices against available options
- Can output to file or stdout
- Provides warnings for missing benchmark configs
- Optionally validates output using `verify_config.py`

**Example usage:**
```bash
# Build config for chat model with vLLM on SLURM
python build_config.py \
    --execution slurm \
    --deployment vllm \
    --export mlflow \
    --model-type chat \
    --benchmarks standard code multilingual \
    --output llama_eval.yaml
```

---

## Implementation Steps

### Execution (2 files)
1. Create `execution/local.yaml`
2. Create `execution/slurm.yaml`

### Deployment (5 files)
3. Create `deployment/none.yaml`
4. Create `deployment/vllm.yaml`
5. Create `deployment/sglang.yaml`
6. Create `deployment/nim.yaml`
7. Create `deployment/trtllm.yaml`

### Export (3 files)
8. Create `export/none.yaml`
9. Create `export/mlflow.yaml`
10. Create `export/wandb.yaml`

### Base Model Evaluation (4 files)
11. Create `evaluation/base/default.yaml`
12. Create `evaluation/base/standard.yaml`
13. Create `evaluation/base/code.yaml`
14. Create `evaluation/base/multilingual.yaml`

### Chat Model Evaluation (6 files)
15. Create `evaluation/chat/default.yaml`
16. Create `evaluation/chat/standard.yaml`
17. Create `evaluation/chat/code.yaml`
18. Create `evaluation/chat/math_reasoning.yaml`
19. Create `evaluation/chat/safety.yaml`
20. Create `evaluation/chat/multilingual.yaml`

### Reasoning Model Evaluation (6 files)
21. Create `evaluation/reasoning/default.yaml`
22. Create `evaluation/reasoning/standard.yaml`
23. Create `evaluation/reasoning/code.yaml`
24. Create `evaluation/reasoning/math_reasoning.yaml`
25. Create `evaluation/reasoning/safety.yaml`
26. Create `evaluation/reasoning/multilingual.yaml`

### Config Builder Script (1 file)
27. Create `.claude/skills/nel-config-generator/scripts/build_config.py`

**Total: 27 files (26 YAML configs + 1 Python script)**

## Verification
- Verify files exist in correct directories
- Validate YAML syntax is correct
- Ensure task names use correct endpoint type (completions vs chat)
- **Validate generated configs** with existing `verify_config.py`:
  ```bash
  python .claude/skills/nel-config-generator/scripts/verify_config.py <generated_config.yaml>
  ```
- Test with `nel run --config <file> --dry-run`

## Source Files Referenced
- `examples/local_basic.yaml` - Local + external API pattern
- `examples/slurm_vllm_basic.yaml` - SLURM + vLLM pattern
- `examples/local_nim.yaml` - NIM deployment pattern
- `examples/slurm_nim.yaml` - SLURM + NIM pattern
- `examples/slurm_sglang_auto_export.yaml` - SGLang + MLflow pattern
- `examples/local_auto_export.yaml` - MLflow export pattern
- `examples/local_reasoning.yaml` - Reasoning model pattern (system prompt config)
- `examples/local_safety.yaml` - Safety benchmarks (Garak, Aegis)
- `examples/slurm_nemotron_benchmarks.yaml` - Reasoning benchmarks with thinking mode
- `examples/local_extra_params.yaml` - Code eval with pass@k (MBPP)
- `src/nemo_evaluator_launcher/configs/deployment/trtllm.yaml` - TRT-LLM config schema
- `src/nemo_evaluator_launcher/exporters/wandb.py` - W&B export config fields
- `docs/evaluation/run-evals/reasoning.md` - Reasoning model evaluation guide
- `docs/evaluation/run-evals/logprobs.md` - Log-probability evaluation guide (base models)
- `examples/local_vllm_logprobs.yaml` - Log-prob evaluation example with tokenizer config

## Task Endpoint Reference
Run `nel ls tasks` to see task endpoint types:
- **completions**: gsm8k, mmlu, gpqa, arc_challenge, humaneval (base), mbpp (base), etc.
- **chat**: ifeval, gpqa_diamond, mmlu_pro, humaneval_instruct, mbpp_plus, garak, aegis_v2, aime_*, livecodebench_*, etc.

---

## Prefab vs Example Comparison Results

All 11 configs built and validated successfully with `build_config.py -v`.

### Summary Table

| # | Example | Status | Key Differences |
|---|---------|--------|-----------------|
| 1 | local_basic.yaml | ✓ | Tasks differ (prefab: 4, example: 3) |
| 2 | local_nim.yaml | ✓ | Tasks differ, parallelism 64 vs 4 |
| 3 | local_vllm_logprobs.yaml | ✓ | Tasks: 8 vs 1 (demo), parallelism differs |
| 4 | local_reasoning.yaml | ✓ | Task names differ (gpqa_diamond_nemo vs simple_evals.mmlu_pro) |
| 5 | local_safety.yaml | ✓ GOOD MATCH | Same tasks (garak, aegis_v2) |
| 6 | local_auto_export.yaml | ✓ | Tasks differ, mlflow config simpler |
| 7 | slurm_vllm_basic.yaml | ✓ | Tasks differ, data_parallel_size 1 vs 8 |
| 8 | slurm_vllm_checkpoint_path.yaml | ✓ | Same build as #7, shows checkpoint_path pattern |
| 9 | slurm_nim.yaml | ✓ | Tasks differ, built adds mount_home: false |
| 10 | slurm_sglang_auto_export.yaml | ✓ | Tasks differ, mlflow config simpler |
| 11 | slurm_nemotron_benchmarks.yaml | ✓ FIXED | No duplicates, uses `process_reasoning_traces: true` |

### Issues Found & Fixed

1. **FIXED: Duplicate gpqa_diamond_nemo** in built config for reasoning + math_reasoning + code benchmarks
   - Removed `gpqa_diamond_nemo` from `reasoning/standard.yaml` (kept in `math_reasoning.yaml`)
   - Added comment explaining the change

2. **FIXED: Added duplicate task validation** to `verify_config.py`
   - New `validate_no_duplicate_tasks` validator in `EvaluationSectionConfig`
   - Added test case `tests/skill_tests/invalid_configs/20_duplicate_tasks.yaml`

3. **Config key decision**: Keeping `process_reasoning_traces: true` (both are valid, this is more explicit)

### Accepted Differences (Per Plan)
- Task lists differ (prefabs have fuller suites, examples are minimal demos)
- Parallelism values differ (prefabs optimized, examples conservative)
- Examples have actual model values, prefabs have `???` placeholders
- MLflow export: examples show full options, prefabs have minimal defaults
