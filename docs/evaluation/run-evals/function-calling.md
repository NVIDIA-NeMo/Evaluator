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

- **Chat Model Deployment**: Function calling requires chat-formatted endpoints
- **BFCL Package**: Berkeley Function Calling Leaderboard evaluation harness
- **Structured Output Support**: Model capable of generating JSON/function call formats

---

## Installation

Install the BFCL evaluation package:

```bash
pip install nvidia-bfcl==25.7.1
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

### AST-Based Function Calling

The most comprehensive BFCL task that evaluates structured function calling:

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

# Configure chat endpoint for function calling
model_name = "megatron_model"
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url=chat_url, 
        type=EndpointType.CHAT, 
        model_id=model_name
    )
)

# Configure BFCL evaluation
eval_config = EvaluationConfig(
    type="bfclv3_ast_prompting", 
    output_dir="/results/function_calling/", 
    params=ConfigParams(limit_samples=10)  # Remove for full dataset
)

# Run evaluation
results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
print(results)
```

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
