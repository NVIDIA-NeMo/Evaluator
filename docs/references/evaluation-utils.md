# Evaluation Utilities Reference

Complete reference for evaluation discovery and utility functions in NeMo Eval.

## nemo_eval.utils.base.list_available_evaluations()

Discovers all pre-defined evaluation configurations across installed evaluation frameworks.

### Function Signature

```python
def list_available_evaluations() -> dict[str, list[str]]
```

### Returns

| Type | Description |
|------|-------------|
| `dict[str, list[str]]` | Dictionary mapping framework names to available task lists |

### Usage Examples

#### Basic Task Discovery

```python
from nemo_eval.utils.base import list_available_evaluations

# Get all available evaluations
available_evals = list_available_evaluations()
print(available_evals)

# Example output:
# {
#     'core_evals.lm_evaluation_harness': [
#         'mmlu', 'gsm8k', 'arc_challenge', 'hellaswag'
#     ],
#     'core_evals.simple_evals': [
#         'AIME_2025', 'humaneval', 'drop'  
#     ],
#     'core_evals.bigcode': [
#         'mbpp', 'humaneval', 'apps'
#     ]
# }
```

#### Framework-Specific Task Listing

```python
available_evals = list_available_evaluations()

# List lm-evaluation-harness tasks
lm_eval_tasks = available_evals.get('core_evals.lm_evaluation_harness', [])
print(f"LM Eval tasks: {lm_eval_tasks}")

# List code generation tasks  
code_tasks = available_evals.get('core_evals.bigcode', [])
print(f"Code tasks: {code_tasks}")
```

#### Comprehensive Task Overview

```python
available_evals = list_available_evaluations()

print(" Available Evaluation Frameworks:")
for framework, tasks in available_evals.items():
    framework_name = framework.replace('core_evals.', '').replace('_', '-')
    print(f"\n {framework_name}:")
    print(f"   Tasks: {len(tasks)}")
    print(f"   Examples: {tasks[:5]}")  # Show first 5 tasks
```

### Error Handling

```python
try:
    evals = list_available_evaluations()
except ImportError as e:
    print(" core_evals package not installed")
    print("Install with: pip install nvidia-lm-eval")
except Exception as e:
    print(f" Error discovering evaluations: {e}")
```

---

## nemo_eval.utils.base.find_framework()

Finds the evaluation framework that implements a specific task.

### Function Signature

```python
def find_framework(eval_task: str) -> str
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `eval_task` | `str` | Name of the evaluation task |

### Returns

| Type | Description |
|------|-------------|
| `str` | Framework name that implements the task |

### Usage Examples

#### Single Task Framework Discovery

```python
from nemo_eval.utils.base import find_framework

# Find framework for MMLU
framework = find_framework("mmlu")
print(f"MMLU is implemented by: {framework}")
# Output: core_evals.lm_evaluation_harness

# Find framework for code evaluation
framework = find_framework("mbpp")  
print(f"MBPP is implemented by: {framework}")
# Output: core_evals.bigcode
```

#### Task Conflict Resolution

```python
# When multiple frameworks implement the same task
try:
    framework = find_framework("mmlu")
    print(f" Found unique framework: {framework}")
except ValueError as e:
    print(f" Multiple frameworks found: {e}")
    # Use explicit framework specification: "lm-evaluation-harness.mmlu"
```

#### Safe Framework Discovery

```python
def safe_find_framework(task: str) -> str:
    """Safely find framework with error handling."""
    try:
        return find_framework(task)
    except ValueError as e:
        if "Multiple frameworks" in str(e):
            print(f" Task '{task}' found in multiple frameworks")
            print("Use format: <framework>.<task>")
            return None
        elif "not found" in str(e):
            print(f" Task '{task}' not available")
            return None
    except Exception as e:
        print(f" Error finding framework: {e}")
        return None

# Usage
framework = safe_find_framework("unknown_task")
```

---

## Health Checking Functions

### nemo_eval.utils.base.wait_for_fastapi_server()

Waits for deployed model server to be ready for evaluation requests.

### Function Signature

```python
def wait_for_fastapi_server(
    base_url: str = "http://0.0.0.0:8080",
    model_name: str = "megatron_model",
    max_retries: int = 600,
    retry_interval: int = 10,
) -> bool
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `base_url` | `str` | `"http://0.0.0.0:8080"` | Base URL of the FastAPI server |
| `model_name` | `str` | `"megatron_model"` | Name of the deployed model |
| `max_retries` | `int` | `600` | Maximum retry attempts |
| `retry_interval` | `int` | `10` | Seconds between retries |

### Returns

| Type | Description |
|------|-------------|
| `bool` | `True` if server is ready, `False` if timeout |

### Usage Examples

#### Basic Health Check

```python
from nemo_eval.utils.base import wait_for_fastapi_server

# Wait for default server
ready = wait_for_fastapi_server()
if ready:
    print(" Server is ready for evaluation")
else:
    print(" Server failed to start")
```

#### Custom Server Configuration

```python
# Wait for custom server configuration
ready = wait_for_fastapi_server(
    base_url="http://my-server:8081",
    model_name="my_custom_model", 
    max_retries=300,
    retry_interval=5
)
```

#### Production Deployment Check

```python
import time

def robust_server_check(base_url: str, model_name: str) -> bool:
    """Robust server health check with logging."""
    print(f" Checking server health: {base_url}")
    start_time = time.time()
    
    ready = wait_for_fastapi_server(
        base_url=base_url,
        model_name=model_name,
        max_retries=180,  # 30 minutes
        retry_interval=10
    )
    
    elapsed = time.time() - start_time
    if ready:
        print(f" Server ready in {elapsed:.1f} seconds")
        return True
    else:
        print(f" Server failed to start after {elapsed:.1f} seconds")
        return False
```

---

## Endpoint Testing Functions

### nemo_eval.utils.base.check_endpoint()

Tests if a specific endpoint is responsive and ready for requests.

### Function Signature

```python
def check_endpoint(
    endpoint_url: str, 
    endpoint_type: str, 
    model_name: str, 
    max_retries: int = 600, 
    retry_interval: int = 2
) -> bool
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `endpoint_url` | `str` | Full URL to the endpoint |
| `endpoint_type` | `str` | Type: `"completions"` or `"chat"` |
| `model_name` | `str` | Model identifier for requests |
| `max_retries` | `int` | Maximum retry attempts |
| `retry_interval` | `int` | Seconds between retries |

### Usage Examples

#### Test Completions Endpoint

```python
from nemo_eval.utils.base import check_endpoint

# Test completions endpoint
ready = check_endpoint(
    endpoint_url="http://0.0.0.0:8080/v1/completions/",
    endpoint_type="completions",
    model_name="megatron_model"
)
```

#### Test Chat Endpoint

```python
# Test chat endpoint
ready = check_endpoint(
    endpoint_url="http://0.0.0.0:8080/v1/chat/completions/",
    endpoint_type="chat", 
    model_name="megatron_model"
)
```

#### Multi-Endpoint Validation

```python
def validate_all_endpoints(base_url: str, model_name: str) -> dict:
    """Validate both completion and chat endpoints."""
    endpoints = {
        "completions": f"{base_url}/v1/completions/",
        "chat": f"{base_url}/v1/chat/completions/"
    }
    
    results = {}
    for endpoint_type, url in endpoints.items():
        print(f" Testing {endpoint_type} endpoint...")
        ready = check_endpoint(
            endpoint_url=url,
            endpoint_type=endpoint_type,
            model_name=model_name,
            max_retries=30,
            retry_interval=2
        )
        results[endpoint_type] = ready
        status = " Ready" if ready else " Failed"
        print(f"   {status}")
    
    return results

# Usage
results = validate_all_endpoints("http://0.0.0.0:8080", "megatron_model")
```

---

## Evaluation Discovery Patterns

### Find All Tasks for a Domain

```python
def find_domain_tasks(domain: str) -> dict:
    """Find all tasks related to a specific domain."""
    available_evals = list_available_evaluations()
    domain_tasks = {}
    
    for framework, tasks in available_evals.items():
        matching_tasks = [
            task for task in tasks 
            if domain.lower() in task.lower()
        ]
        if matching_tasks:
            domain_tasks[framework] = matching_tasks
    
    return domain_tasks

# Find all math-related tasks
math_tasks = find_domain_tasks("math")
print(f"Math evaluation tasks: {math_tasks}")

# Find code-related tasks  
code_tasks = find_domain_tasks("code")
print(f"Code evaluation tasks: {code_tasks}")
```

### Framework Capability Matrix

```python
def create_capability_matrix() -> dict:
    """Create a matrix of framework capabilities."""
    available_evals = list_available_evaluations()
    
    capabilities = {
        "academic": ["mmlu", "arc", "hellaswag"],
        "reasoning": ["gsm8k", "math", "reasoning"],
        "coding": ["humaneval", "mbpp", "apps"],
        "safety": ["safety", "toxic", "bias"],
        "multilingual": ["multilingual", "translation"]
    }
    
    matrix = {}
    for capability, keywords in capabilities.items():
        matrix[capability] = {}
        for framework, tasks in available_evals.items():
            matching_tasks = []
            for keyword in keywords:
                matching_tasks.extend([
                    task for task in tasks 
                    if keyword in task.lower()
                ])
            if matching_tasks:
                framework_name = framework.replace('core_evals.', '')
                matrix[capability][framework_name] = matching_tasks
    
    return matrix

# Usage
matrix = create_capability_matrix()
for capability, frameworks in matrix.items():
    print(f"\n {capability.title()} Evaluation:")
    for framework, tasks in frameworks.items():
        print(f"   {framework}: {len(tasks)} tasks")
```

---

**Source**: `src/nemo_eval/utils/base.py:110-156`  
**Evidence**: Complete implementations of evaluation discovery and health checking functions
