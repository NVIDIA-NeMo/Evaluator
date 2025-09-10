(function-calling)=

# Function Calling Evaluation

Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors using the Berkeley Function Calling Leaderboard (BFCL).

## Overview

Function calling evaluation measures a model's ability to:

- **Tool Discovery**: Identify appropriate functions for given tasks
- **Parameter Extraction**: Extract correct parameters from natural language
- **API Integration**: Generate proper function calls and handle responses  
- **Multi-Step Reasoning**: Chain function calls for complex workflows
- **Error Handling**: Manage invalid parameters and API failures

## Before You Start

Ensure you have:

- **Chat Model Endpoint**: Function calling requires chat-formatted OpenAI-compatible endpoints
- **API Access**: Valid API key for your model endpoint
- **Structured Output Support**: Model capable of generating JSON/function call formats

---

## Choose Your Approach

::::{tab-set}
:::{tab-item} NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run function calling evaluations with unified CLI:

```bash
# List available function calling tasks
nv-eval ls tasks | grep -E "(bfcl|function)"

# Run BFCL AST prompting evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["bfclv3_ast_prompting"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}

# Run multiple function calling benchmarks
nv-eval run \
    --config-dir examples \
    --config-name local_function_calling_suite \
    -o 'evaluation.tasks=["bfclv3_simple", "bfclv3_ast_prompting", "bfclv3_parallel"]'
```
:::

:::{tab-item}  Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams
)

# Configure function calling evaluation
eval_config = EvaluationConfig(
    type="bfclv3_ast_prompting",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,    # Remove for full dataset
        temperature=0.1,     # Low temperature for precise function calls
        max_new_tokens=512,  # Adequate for function call generation
        top_p=0.9
    )
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct", 
        type="chat",
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
# Pull and run BFCL evaluation container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/bfcl:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here

# Run function calling evaluation
eval-factory run_eval \
    --eval_type bfclv3_ast_prompting \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10,config.params.temperature=0.1'
```
:::
::::

## Installation

Install the BFCL evaluation package for local development:

```bash
pip install nvidia-bfcl==25.7.1
```

## Discovering Available Tasks

Use the launcher CLI to discover all available function calling tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter for function calling tasks
nv-eval ls tasks | grep -E "(bfcl|function)"

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task bfclv3_ast_prompting
```

## Available Function Calling Tasks

BFCL provides comprehensive function calling benchmarks:

| Task | Description | Complexity | Format |
|------|-------------|------------|---------|
| `bfclv3_ast_prompting` | AST-based function calling | Intermediate | Structured |
| `bfclv3_exec_prompting` | Executable function calls | Advanced | Code execution |
| `bfclv3_simple` | Basic function invocation | Beginner | Simple API calls |
| `bfclv3_parallel` | Multi-function orchestration | Advanced | Parallel execution |

## Basic Function Calling Evaluation

The most comprehensive BFCL task is AST-based function calling that evaluates structured function calling. Use any of the three approaches above to run BFCL evaluations.

### Understanding Function Calling Format

BFCL evaluates models on their ability to generate proper function calls:

**Input Example**:
```
What's the weather like in San Francisco and New York?

Available functions:
- get_weather(city: str, units: str = "celsius") -> dict
```

**Expected Output**:
```json
[
    {"name": "get_weather", "arguments": {"city": "San Francisco"}},
    {"name": "get_weather", "arguments": {"city": "New York"}}
]
```

## Advanced Configuration

### Custom Evaluation Parameters

```python
# Optimized settings for function calling
eval_params = ConfigParams(
    limit_samples=100,
    parallelism=2,              # Conservative for complex reasoning
    temperature=0.1,            # Low temperature for precise function calls
    max_tokens=512,             # Adequate for function call generation
    top_p=0.9                   # Focused sampling for accuracy
)

eval_config = EvaluationConfig(
    type="bfclv3_ast_prompting",
    output_dir="/results/bfcl_optimized/",
    params=eval_params
)
```

### Multi-Task Function Calling Evaluation

Evaluate across different function calling scenarios:

```python
function_calling_tasks = [
    "bfclv3_simple",        # Basic function calls
    "bfclv3_ast_prompting", # Structured calls  
    "bfclv3_parallel"       # Multi-function coordination
]

results = {}

for task in function_calling_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/{task}/",
        params=ConfigParams(
            limit_samples=50,
            temperature=0.0,    # Deterministic for consistency
            parallelism=1       # Sequential for complex reasoning
        )
    )
    
    results[task] = evaluate(
        target_cfg=target_config, 
        eval_cfg=eval_config
    )
    
    print(f"{task}: {results[task]['overall_accuracy']:.2%}")
```

## Understanding Metrics

### Function Calling Accuracy Types

```python
# Example BFCL results breakdown
{
    "overall_accuracy": 0.78,           # Overall function calling success
    "parameter_accuracy": 0.85,         # Correct parameter extraction  
    "function_selection": 0.82,         # Appropriate function choice
    "format_compliance": 0.91,          # Valid JSON/call format
    "multi_turn_success": 0.71          # Complex conversation handling
}
```

### Performance Benchmarks

| Model Category | BFCL Score | Function Selection | Parameter Accuracy |
|----------------|------------|-------------------|-------------------|
| Function-Tuned | 80-90% | 90-95% | 85-92% |
| Chat-Optimized | 65-80% | 75-85% | 70-85% |
| Base Chat | 40-65% | 60-75% | 50-70% |

## Common Function Calling Patterns

### Single Function Calls

Simple, direct function invocation for straightforward requests.

### Parallel Function Execution

Multiple independent function calls for complex queries:
```python
# User: "Get weather for London and Tokyo, and current time in both"
# Expected: parallel calls to get_weather() and get_time()
```

### Sequential Function Chaining

Using output from one function as input to another:
```python
# User: "Find restaurants near the Eiffel Tower"
# Expected: get_location("Eiffel Tower") → find_restaurants(coordinates)
```

## Integration with Agent Frameworks

Function calling evaluation results help assess model readiness for:

- **LangChain Integration**: Tool calling capabilities
- **AutoGPT Systems**: Multi-step task automation
- **Custom Agent Frameworks**: API integration accuracy
- **Workflow Automation**: Reliable function orchestration

---

*For more function calling tasks and advanced configurations, see the [BFCL package documentation](https://pypi.org/project/nvidia-bfcl/).*
