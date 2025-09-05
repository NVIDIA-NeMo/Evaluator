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

- **Model Deployed**: A chat-enabled model deployment via [PyTriton](../../deployment/pytriton.md) or [Ray Serve](../../deployment/ray-serve.md)
- **BigCode Package**: Install the BigCode evaluation harness
- **Sufficient Context**: Models with adequate context length for code problems

---

## Installation

Install the BigCode evaluation package:

```bash
pip install nvidia-bigcode-eval==25.7.1
```

## Available Tasks

The BigCode harness provides various programming benchmarks:

| Task | Description | Language | Difficulty |
|------|-------------|----------|------------|
| `mbpp` | Mostly Basic Programming Problems | Python | Beginner-Intermediate |
| `humaneval` | Hand-written programming problems | Python | Intermediate |
| `apps` | Competitive programming problems | Python | Advanced |
| `codegen` | Code generation from docstrings | Python | Intermediate |

## Basic Code Generation Evaluation

### Python Programming (MBPP)

The Most Basic Programming Problems (MBPP) benchmark tests fundamental programming skills:

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

# Configure endpoint for chat-based evaluation
model_name = "megatron_model"
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url=chat_url, 
        type=EndpointType.CHAT, 
        model_id=model_name
    )
)

# Configure MBPP evaluation
eval_config = EvaluationConfig(
    type="mbpp", 
    output_dir="/results/", 
    params=ConfigParams(limit_samples=10)  # Remove for full dataset
)

# Run evaluation
results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
print(results)
```

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
    max_tokens=1024,           # Sufficient tokens for complete functions
    stop_sequences=["```"]      # Stop at code block end
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
code_tasks = ["mbpp", "humaneval", "apps"]
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
