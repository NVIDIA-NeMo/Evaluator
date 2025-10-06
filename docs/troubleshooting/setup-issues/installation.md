(installation-issues)=

# Installation Issues

Solutions for import errors, missing dependencies, and framework installation problems.

## Common Import and Installation Problems

###  Problem: `ModuleNotFoundError: No module named 'core_evals'`

**Solution**:
```bash
# Install missing core evaluation framework
pip install nvidia-lm-eval

# For additional frameworks
pip install nvidia-simple-evals nvidia-bigcode-eval nvidia-bfcl
```

###  Problem: `Framework for task X not found`

**Diagnosis**:
```python
from nemo_evaluator import show_available_tasks

# Display all available tasks
print("Available tasks:")
show_available_tasks()
```

Or use the CLI:
```bash
nv-eval ls tasks
```

**Solution**:
```bash
# Install the framework containing the missing task
pip install nvidia-<framework-name>

# Restart Python session to reload frameworks
```

###  Problem: `Multiple frameworks found for task X`

**Solution**:
```python
# Use explicit framework specification
config = EvaluationConfig(
    type="lm-evaluation-harness.mmlu",  # Instead of just "mmlu"
    # ... other config
)
```

