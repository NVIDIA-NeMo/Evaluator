# Debug and Troubleshoot Advanced Issues

Comprehensive debugging techniques, monitoring strategies, and common issue patterns for complex evaluation problems.

## Advanced Debugging Approach

When basic troubleshooting doesn't resolve your issue, use these systematic debugging techniques:

::::{tab-set}

:::{tab-item} Comprehensive Health Check

```python
from nemo_evaluator import show_available_tasks
import requests
import torch

def full_system_check():
    """Complete system health check"""
    print("🔍 NeMo Eval System Health Check")
    print("=" * 40)
    
    # Framework availability
    try:
        frameworks = list_available_evaluations()
        print(f"✅ Available frameworks: {len(frameworks)}")
    except Exception as e:
        print(f"❌ Framework check failed: {e}")
    
    # Server connectivity
    try:
        ready = wait_for_fastapi_server("http://0.0.0.0:8080", max_retries=3)
        print(f"✅ Server ready: {ready}")
    except Exception as e:
        print(f"❌ Server check failed: {e}")
    
    # GPU availability
    if torch.cuda.is_available():
        print(f"✅ GPU count: {torch.cuda.device_count()}")
    else:
        print("⚠️  No CUDA GPUs available")

full_system_check()
```

:::

:::{tab-item} Detailed Logging

```python
import logging

# Enable comprehensive logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("nemo_evaluator").setLevel(logging.DEBUG)

# Monitor evaluation progress
def log_evaluation_progress():
    import psutil
    import time
    
    start_time = time.time()
    while True:  # Until evaluation completes
        cpu = psutil.cpu_percent()
        memory = psutil.virtual_memory().percent
        elapsed = time.time() - start_time
        print(f"⏱️  T+{elapsed:.0f}s | CPU: {cpu:.1f}% | Memory: {memory:.1f}%")
        time.sleep(10)
```

:::

:::{tab-item} Minimal Test Case

```python
def minimal_evaluation_test():
    """Simplest possible evaluation for debugging"""
    from nemo_evaluator import evaluate
    from nemo_evaluator.api.api_dataclasses import *
    
    # Absolute minimal configuration
    api_endpoint = ApiEndpoint(
        url="http://0.0.0.0:8080/v1/completions/",
        type="completions",
        model_id="megatron_model"
    )
    
    target = EvaluationTarget(api_endpoint=api_endpoint)
    config = EvaluationConfig(
        type="gsm8k",
        params=ConfigParams(limit_samples=1, parallelism=1, temperature=0)
    )
    
    try:
        result = evaluate(target_cfg=target, eval_cfg=config)
        print("✅ Minimal evaluation successful")
        return True
    except Exception as e:
        print(f"❌ Minimal evaluation failed: {e}")
        return False

minimal_evaluation_test()
```

:::

::::

## Advanced Categories

Deep-dive resources for complex troubleshooting scenarios:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Debugging Techniques
:link: debugging-guide
:link-type: doc

Systematic debugging approaches, monitoring strategies, and error recovery patterns.
:::

:::{grid-item-card} {octicon}`pattern;1.5em;sd-mr-1` Common Patterns
:link: common-patterns
:link-type: doc

Frequently encountered problems with solutions based on actual test patterns and user reports.
:::

::::

## When to Use Advanced Debugging

Use these techniques when:

- **Basic troubleshooting fails**: Standard solutions don't resolve the issue
- **Intermittent failures**: Problems occur sporadically or under specific conditions
- **Performance degradation**: Evaluations work but are slower than expected
- **Complex environments**: Multi-GPU, cluster, or containerized deployments
- **Custom configurations**: Non-standard setups or custom evaluation frameworks

## Expert-Level Troubleshooting Process

1. **Systematic Isolation**: Test each component (endpoint, config, task) independently
2. **Progressive Complexity**: Start minimal and gradually add complexity
3. **Resource Monitoring**: Track system resources throughout the debugging process
4. **Log Analysis**: Enable detailed logging and analyze patterns in failures
5. **Reproducible Test Cases**: Create minimal examples that consistently reproduce issues
6. **Environment Documentation**: Document exact versions, configurations, and environment details

For immediate help with common issues, check {doc}`../setup-issues/index` and {doc}`../runtime-issues/index` first.

:::{toctree}
:caption: Advanced Debugging
:hidden:

Debugging Guide <debugging-guide>
Common Patterns <common-patterns>
:::
