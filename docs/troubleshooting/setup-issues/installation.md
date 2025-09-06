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
pip install nvidia-simple-evals nvidia-bigcode nvidia-bfcl
```

###  Problem: `Framework for task X not found`

**Diagnosis**:
```python
from nemo_eval.utils.base import list_available_evaluations
tasks = list_available_evaluations()
print("Available tasks:", [t for task_list in tasks.values() for t in task_list])
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

## Prevention Tips

1. **Install all evaluation frameworks** at once to avoid missing dependencies:
```bash
pip install nvidia-lm-eval nvidia-simple-evals nvidia-bigcode nvidia-bfcl
```

2. **Restart your Python session** after installing new frameworks to ensure they're properly loaded.

3. **Use explicit framework names** in task specifications to avoid conflicts:
   - `lm-evaluation-harness.mmlu` instead of just `mmlu`
   - `simple-evals.hellaswag` instead of just `hellaswag`

4. **Check available tasks** before running evaluations to confirm framework installation.
