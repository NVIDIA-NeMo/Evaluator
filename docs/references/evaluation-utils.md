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

This function scans all installed `core_evals` packages and prints a hierarchical list of available evaluation tasks organized by framework. It is useful for discovering which benchmarks and tasks are available in your environment.

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

For programmatic access to task information, use the CLI or launcher API:

```bash
# Using CLI
nv-eval ls tasks

# Filter for specific tasks
nv-eval ls tasks | grep mmlu
```

```python
# Using Launcher API
from nemo_evaluator_launcher.api.functional import get_tasks_list

# Get structured task information
tasks = get_tasks_list()
for task in tasks:
    task_name, endpoint_type, harness, container = task
    print(f"Task: {task_name}, Type: {endpoint_type}, Framework: {harness}")
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

The function handles missing packages gracefully:

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

## Task Discovery Patterns

### Find Tasks by Domain

Use the CLI for domain-specific task discovery:

```bash
# Find all math-related tasks
nv-eval ls tasks | grep -i math

# Find code generation tasks
nv-eval ls tasks | grep -iE "(code|humaneval|mbpp)"

# Find safety evaluation tasks
nv-eval ls tasks | grep -iE "(safety|toxic|bias)"
```

### Framework Capability Matrix

Different frameworks specialize in different evaluation types:

| Framework | Primary Use Case | Example Tasks |
|-----------|-----------------|---------------|
| **lm-evaluation-harness** | Academic benchmarks, reasoning | MMLU, GSM8K, ARC, HellaSwag |
| **simple-evals** | Specialized reasoning, competition problems | AIME, MATH, MGSM, MMLU-Pro |
| **bigcode** | Code generation and understanding | HumanEval, MBPP, APPS |
| **BFCL** | Function calling, tool use | AST prompting, executable calls |

### Discover Tasks for Specific Endpoints

Different tasks require different endpoint types:

```bash
# Chat endpoint tasks (instruction-following)
nv-eval ls tasks | awk '$2 == "chat"'

# Completions endpoint tasks (base models)
nv-eval ls tasks | awk '$2 == "completions"'

# Vision-language tasks
nv-eval ls tasks | awk '$2 == "vlm"'
```

---

## Integration with Evaluation Workflows

### Pre-Flight Task Verification

Verify task availability before running evaluations:

```python
from nemo_evaluator import show_available_tasks
import subprocess

def verify_task_available(task_name: str) -> bool:
    """Check if a specific task is available."""
    result = subprocess.run(
        ["nv-eval", "ls", "tasks"],
        capture_output=True,
        text=True
    )
    return task_name in result.stdout

# Usage
if verify_task_available("mmlu"):
    print("✓ MMLU is available")
else:
    print("✗ MMLU not found. Install: pip install nvidia-lm-eval")
```

### Dynamic Task Configuration

Use task discovery for dynamic configuration:

```python
from nemo_evaluator_launcher.api.functional import get_tasks_list
from nemo_evaluator_launcher.api.types import RunConfig

# Get all chat endpoint tasks
tasks = get_tasks_list()
chat_tasks = [t[0] for t in tasks if t[1] == "chat"]

# Run evaluations for all chat tasks
for task in chat_tasks[:5]:  # Run first five
    config = RunConfig.from_hydra(
        config_dir="examples",
        config_name="local_llama_3_1_8b_instruct",
        hydra_overrides=[f"evaluation.tasks=[{task}]"]
    )
    # Execute evaluation...
```

### Framework Selection

When multiple frameworks provide the same task, use explicit framework specification:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig

# Explicit framework specification
config = EvaluationConfig(
    type="lm-evaluation-harness.mmlu",  # Instead of just "mmlu"
    # ... other configuration
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

**Diagnosis**:
```python
# Check if package is installed
import importlib
try:
    importlib.import_module("core_evals.lm_evaluation_harness")
    print("✓ LM Eval Harness installed")
except ImportError:
    print("✗ LM Eval Harness not installed")
```

**Note**: The module name `lm_evaluation_harness` represents the internal package structure and may vary based on the framework package organization.

**Solution**:
```bash
# Install missing framework
pip install nvidia-lm-eval

# Restart Python session to reload packages
```

### Problem: Task conflicts between frameworks

When multiple frameworks implement the same task name (for example, both `lm-evaluation-harness` and `simple-evals` may provide `mmlu`), use explicit framework specification:

**Solution**:
```bash
# CLI approach - use explicit framework.task format
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["lm-evaluation-harness.mmlu"]'

# Alternative: check which frameworks provide a task
nv-eval ls tasks | grep mmlu
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
nv-eval ls tasks

# List recent evaluation runs
nv-eval ls runs

# Get detailed help
nv-eval --help
```

---

**Source**: `packages/nemo-evaluator/src/nemo_evaluator/core/entrypoint.py:105-123`  
**API Export**: `nemo_evaluator/__init__.py` exports `show_available_tasks` for public use  
**Related**: See {ref}`quickstart-guide` for evaluation setup and {ref}`benchmarks-overview` for task descriptions
