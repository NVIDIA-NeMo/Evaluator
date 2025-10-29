---
orphan: true
---

(evaluation-utils-reference)=

# Evaluation Utilities Reference

Complete reference for evaluation discovery and utility functions in NeMo Evaluator.

## nemo_evaluator.show_available_tasks()

Discovers and displays all available evaluation tasks across installed evaluation frameworks.

### Function Signature

```python
def show_available_tasks() -> None
```

### Returns

| Type | Description |
|------|-------------|
| `None` | Prints available tasks to stdout |

### Description

This function scans all installed `core_evals` packages and prints a hierarchical list of available evaluation tasks organized by framework. Use this function to discover which benchmarks and tasks are available in your environment.

The function automatically detects:

- **Installed frameworks**: lm-evaluation-harness, simple-evals, bigcode, BFCL
- **Available tasks**: All tasks defined in each framework's configuration
- **Installation status**: Displays message if no evaluation packages are installed

### Usage Examples

#### Basic Task Discovery

```python
from nemo_evaluator import show_available_tasks

# Display all available evaluations
show_available_tasks()

# Example output:
# lm-evaluation-harness: 
#   * mmlu
#   * gsm8k
#   * arc_challenge
#   * hellaswag
# simple-evals:
#   * AIME_2025
#   * humaneval
#   * drop
# bigcode:
#   * mbpp
#   * humaneval
#   * apps
```

#### Programmatic Task Discovery

For programmatic access to task information, use the launcher API:

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list

# Get structured task information
tasks = get_tasks_list()
for task in tasks:
    task_name, endpoint_type, harness, container = task
    print(f"Task: {task_name}, Type: {endpoint_type}, Framework: {harness}")
```

To filter tasks using the CLI:

```bash
# List all tasks
nemo-evaluator-launcher ls tasks

# Filter for specific tasks
nemo-evaluator-launcher ls tasks | grep mmlu
```

#### Check Installation Status

```python
from nemo_evaluator import show_available_tasks

# Check if evaluation packages are installed
print("Available evaluation frameworks:")
show_available_tasks()

# If no packages installed, you'll see:
# NO evaluation packages are installed.
```

### Installation Requirements

To use this function, install evaluation framework packages:

```bash
# Install all frameworks
pip install nvidia-lm-eval nvidia-simple-evals nvidia-bigcode-eval nvidia-bfcl

# Or install selectively
pip install nvidia-lm-eval        # LM Evaluation Harness
pip install nvidia-simple-evals   # Simple Evals
pip install nvidia-bigcode-eval   # BigCode benchmarks
pip install nvidia-bfcl           # Berkeley Function Calling Leaderboard
```

### Error Handling

The function handles missing packages:

```python
from nemo_evaluator import show_available_tasks

# Safely check for available tasks
try:
    show_available_tasks()
except ImportError as e:
    print(f"Error: {e}")
    print("Install evaluation frameworks: pip install nvidia-lm-eval")
```

---

## Integration with Evaluation Workflows

### Pre-Flight Task Verification

Verify task availability before running evaluations:

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list

def verify_task_available(task_name: str) -> bool:
    """Check if a specific task is available."""
    tasks = get_tasks_list()
    return any(task[0] == task_name for task in tasks)

# Usage
if verify_task_available("mmlu"):
    print("✓ MMLU is available")
else:
    print("✗ MMLU not found. Install evaluation framework packages")
```

### Filter Tasks by Endpoint Type

Use task discovery to filter by endpoint type:

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list

# Get all chat endpoint tasks
tasks = get_tasks_list()
chat_tasks = [task[0] for task in tasks if task[1] == "chat"]
completions_tasks = [task[0] for task in tasks if task[1] == "completions"]

print(f"Chat tasks: {chat_tasks[:5]}")  # Show first five
print(f"Completions tasks: {completions_tasks[:5]}")
```

### Framework Selection

When a task is provided by more than one framework, use explicit framework specification in your configuration:

```python
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, ConfigParams

# Explicit framework specification
config = EvaluationConfig(
    type="lm-evaluation-harness.mmlu",  # Instead of just "mmlu"
    params=ConfigParams(task="mmlu")
)
```

---

## Troubleshooting

### Problem: "NO evaluation packages are installed"

**Solution**:

```bash
# Install evaluation frameworks
pip install nvidia-lm-eval nvidia-simple-evals nvidia-bigcode-eval nvidia-bfcl

# Verify installation
python -c "from nemo_evaluator import show_available_tasks; show_available_tasks()"
```

### Problem: Task not appearing in list

**Solution**:

```bash
# Install the required framework package
pip install nvidia-lm-eval

# Verify installation
python -c "from nemo_evaluator import show_available_tasks; show_available_tasks()"
```

### Problem: Task conflicts between frameworks

When a task name is provided by more than one framework (for example, both `lm-evaluation-harness` and `simple-evals` provide `mmlu`), use explicit framework specification:

**Solution**:

```bash
# Use explicit framework.task format in your configuration overrides
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["lm-evaluation-harness.mmlu"]'
```

---

## Related Functions

### NeMo Evaluator Launcher API

For programmatic access with structured results:

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list

# Returns list of tuples: (task_name, endpoint_type, framework, container)
tasks = get_tasks_list()
```

### CLI Commands

```bash
# List all tasks
nemo-evaluator-launcher ls tasks

# List recent evaluation runs
nemo-evaluator-launcher ls runs

# Get detailed help
nemo-evaluator-launcher --help
```

---

**Source**: `packages/nemo-evaluator/src/nemo_evaluator/core/entrypoint.py:105-123`  
**API Export**: `nemo_evaluator/__init__.py` exports `show_available_tasks` for public use  
**Related**: See {ref}`gs-quickstart` for evaluation setup and {ref}`eval-benchmarks` for task descriptions
