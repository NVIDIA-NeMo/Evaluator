---
orphan: true
---

# Runtime and Execution Issues

Solutions for problems that occur during evaluation execution, including configuration validation and launcher management.

## Common Runtime Problems

When evaluations fail during execution, start with these diagnostic steps:

::::{tab-set}

:::{tab-item} Configuration Check

```bash
# Validate configuration before running
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct --dry-run

# Test minimal configuration
python -c "
from nemo_evaluator import EvaluationConfig, ConfigParams
config = EvaluationConfig(type='mmlu', params=ConfigParams(limit_samples=1))
print('Configuration valid')
"
```

:::

:::{tab-item} Endpoint Test

```python
import requests

# Test model endpoint connectivity
response = requests.post(
    "http://0.0.0.0:8080/v1/completions/",
    json={"prompt": "test", "model": "megatron_model", "max_tokens": 1}
)
print(f"Endpoint status: {response.status_code}")
```

:::

:::{tab-item} Resource Monitor

```bash
# Monitor system resources during evaluation
nvidia-smi -l 1  # GPU usage
htop            # CPU/Memory usage
```

:::

::::

## Runtime Categories

Choose the category that matches your runtime issue:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Issues
:link: configuration
:link-type: doc

Config parameter validation, tokenizer setup, and endpoint configuration problems.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` Launcher Issues
:link: launcher
:link-type: doc

NeMo Evaluator Launcher-specific problems including job management and multi-backend execution.
:::

::::


:::{toctree}
:caption: Runtime Issues
:hidden:

Configuration <configuration>
Launcher <launcher>
:::
