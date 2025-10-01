(code-generation)=

# Code Generation Evaluation

Evaluate programming capabilities through code generation, completion, and algorithmic problem solving using the BigCode evaluation harness.


## Overview

Code generation evaluation assesses a model's ability to:

- **Generate Code**: Write complete functions from natural language descriptions
- **Code Completion**: Fill in missing code segments 
- **Algorithm Implementation**: Solve programming challenges and competitive programming problems
- **Multi-Language Support**: Evaluate across Python, JavaScript, Java, and other programming languages

## Before You Start

Ensure you have:

- **Model Endpoint**: An OpenAI-compatible endpoint for your model
- **API Access**: Valid API key for your model endpoint
- **Sufficient Context**: Models with adequate context length for code problems

---

## Choose Your Approach

::::{tab-set}
:::{tab-item}  NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run code generation evaluations with unified CLI:

```bash
# List available code generation tasks
nv-eval ls tasks | grep -E "(mbpp|humaneval|apps|codegen)"

# Run MBPP evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["mbpp"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}

# Run multiple code generation benchmarks
nv-eval run \
    --config-dir examples \
    --config-name local_code_generation_suite \
    -o 'evaluation.tasks=["mbpp", "humaneval"]'
```
:::

:::{tab-item}  Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams, EndpointType
)

# Configure code generation evaluation
eval_config = EvaluationConfig(
    type="mbpp",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,    # Remove for full dataset
        temperature=0.2,     # Low temperature for consistent code
        max_new_tokens=1024, # Sufficient tokens for complete functions
        top_p=0.9
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct", 
        type=EndpointType.CHAT,
        api_key="your_api_key"
    )
)

result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```
:::

:::{tab-item}  Containers Directly
:sync: containers

For specialized container workflows:

```bash
# Pull and run BigCode evaluation container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here

# Run code generation evaluation
eval-factory run_eval \
    --eval_type mbpp \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10,config.params.temperature=0.2'
```
:::
::::

## Container Access

The BigCode evaluation harness is available through Docker containers. No separate package installation is required:

```bash
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
```

## Discovering Available Tasks

Use the launcher CLI to discover all available code generation tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter for code generation tasks
nv-eval ls tasks | grep -E "(mbpp|humaneval)"

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task mbpp
```

## Available Tasks

The BigCode harness provides various programming benchmarks:

| Task | Description | Language | Difficulty |
|------|-------------|----------|------------|
| `mbpp` | Mostly Basic Programming Problems | Python | Beginner-Intermediate |
| `mbppplus` | Extended MBPP with additional test cases | Python | Beginner-Intermediate |
| `humaneval` | Hand-written programming problems | Python | Intermediate |

## Basic Code Generation Evaluation

The Most Basic Programming Problems (MBPP) benchmark tests fundamental programming skills. Use any of the three approaches above to run MBPP evaluations.

### Understanding Results

Code generation evaluations typically report:

- **Pass@1**: Percentage of problems solved on first attempt
- **Pass@k**: Percentage of problems solved in k attempts  
- **Execution Rate**: Percentage of generated code that compiles/runs
- **Syntax Accuracy**: Percentage of syntactically correct code

## Advanced Configuration

### Custom Evaluation Parameters

```python
# Advanced configuration for code generation
eval_params = ConfigParams(
    limit_samples=100,           # Evaluate on subset for testing
    parallelism=4,              # Concurrent evaluation requests
    temperature=0.2,            # Low temperature for consistent code
    max_new_tokens=1024         # Sufficient tokens for complete functions
)

eval_config = EvaluationConfig(
    type="mbpp",
    output_dir="/results/mbpp_advanced/",
    params=eval_params
)
```

### Multiple Task Evaluation

Evaluate across different code generation benchmarks:

```python
code_tasks = ["mbpp", "humaneval", "mbppplus"]
results = {}

for task in code_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/{task}/",
        params=ConfigParams(
            limit_samples=50,
            temperature=0.1,
            parallelism=2
        )
    )
    
    results[task] = evaluate(
        target_cfg=target_config, 
        eval_cfg=eval_config
    )
```

## Understanding Metrics

### Pass@k Interpretation

```python
# Pass@k results interpretation
{
    "pass@1": 0.65,    # 65% solved on first attempt
    "pass@5": 0.82,    # 82% solved within 5 attempts  
    "pass@10": 0.89    # 89% solved within 10 attempts
}
```

### Performance Benchmarks

| Model Type | MBPP Pass@1 | HumanEval Pass@1 | Notes |
|------------|-------------|------------------|-------|
| Code-Specialized | 70-85% | 60-75% | Fine-tuned on code |
| General LLM | 40-60% | 30-50% | No code specialization |
| Base Model | 15-30% | 10-25% | Pre-training only |

---

*For more code generation tasks and configurations, see the [BigCode package documentation](https://pypi.org/project/nvidia-bigcode-eval/).*
