(safety-security)=

# Safety and Security Evaluation

Test AI safety, alignment, and security vulnerabilities using specialized safety harnesses and probing techniques to ensure responsible AI deployment.

## Overview

Safety and security evaluation assesses models for:

- **Content Safety**: Detection of harmful, toxic, or inappropriate content generation
- **Alignment Testing**: Adherence to human values and intended behavior patterns
- **Jailbreak Resistance**: Robustness against prompt injection and manipulation attempts
- **Bias Detection**: Identification of demographic, cultural, or social biases
- **Security Vulnerabilities**: Resistance to adversarial attacks and data extraction

## Before You Start

Ensure you have:

- **Model Endpoint**: Chat-enabled OpenAI-compatible endpoint for interactive safety testing
- **API Access**: Valid API key for your model endpoint
- **Judge Model Access**: API access to safety evaluation models (NemoGuard, etc.)
- **Authentication**: Hugging Face token for accessing gated safety datasets

---

## Choose Your Approach

::::{tab-set}
:::{tab-item}  NeMo Evaluator Launcher
:sync: launcher

**Recommended** - The fastest way to run safety & security evaluations with unified CLI:

```bash
# List available safety tasks
nv-eval ls tasks | grep -E "(safety|aegis|garak)"

# Run Aegis safety evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["aegis_v2"]' \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY}

# Run safety and security evaluation
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o 'evaluation.tasks=["aegis_v2", "garak"]'
```
:::

:::{tab-item}  Core API
:sync: api

For programmatic evaluation in custom workflows:

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, ConfigParams
)

# Configure safety evaluation
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,    # Remove for full dataset
        temperature=0.7,     # Natural conversation temperature
        max_new_tokens=512,
        parallelism=1,       # Sequential for safety analysis
        extra={
            "judge": {
                "model_id": "llama-nemotron-safety-guard-v2",
                "url": "http://0.0.0.0:9000/v1/completions",
                "api_key": "your_judge_api_key"
            }
        }
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
# Pull and run Safety Harness container
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/safety-harness:{{ docker_compose_latest }} bash

# Inside container - set environment
export MY_API_KEY=your_api_key_here
export HF_TOKEN=your_hf_token_here

# Run safety evaluation
eval-factory run_eval \
    --eval_type aegis_v2 \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10,config.params.temperature=0.7'

# For security testing with Garak
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/garak:{{ docker_compose_latest }} bash
```
:::
::::

## Installation

Install the safety evaluation packages for local development:

```bash
# Safety harness for comprehensive safety evaluation
pip install nvidia-safety-harness==25.6

# Garak for security vulnerability scanning  
pip install nvidia-eval-factory-garak==25.6
```

## Authentication Setup

Many safety evaluations require external judge models and gated datasets:

```bash
# NVIDIA Build API key for judge models
export JUDGE_API_KEY="your_nvidia_api_key"

# Hugging Face token for gated safety datasets  
export HF_TOKEN="your_hf_token"
```

## Discovering Available Tasks

Use the launcher CLI to discover all available safety and security tasks:

```bash
# List all available benchmarks
nv-eval ls tasks

# Filter for safety and security tasks
nv-eval ls tasks | grep -E "(safety|aegis|garak)"

# Get detailed information about a specific task (if supported)
nv-eval ls tasks --task aegis_v2
```

## Available Safety Tasks

### Safety Harness Tasks

| Task | Description | Judge Model Required | Dataset Access |
|------|-------------|---------------------|----------------|
| `aegis_v2` | Content safety evaluation | NemoGuard 8B | Gated |

### Garak Security Tasks

The `garak` task provides comprehensive security testing through configurable probes:

| Probe Category | Example Probes | Description |
|----------------|----------------|-------------|
| Encoding | `encoding.*` | Test handling of various encodings |
| Prompt Injection | `promptinject.*` | Test resistance to prompt manipulation |
| Data Leakage | `leakage.*` | Test for information extraction vulnerabilities |

Configure specific probes using the `extra.probes` parameter (refer to examples below).

## Basic Safety Evaluation

Content safety evaluation using NVIDIA's NemoGuard safety model can be performed using any of the three approaches above. The safety evaluation requires a separate judge model deployment for scoring responses.

## Advanced Safety Configuration

### Comprehensive Safety Testing

```python
# Aegis V2 safety evaluation with judge model
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="/results/safety/aegis_v2/",
    params=ConfigParams(
        limit_samples=100,
        parallelism=1,          # Sequential for safety analysis
        temperature=0.7,        # Natural conversation temperature
        extra={
            "judge": {
                "model_id": "llama-nemotron-safety-guard-v2",
                "url": "http://0.0.0.0:9000/v1/completions",
            }
        }
    )
)

safety_result = evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config
)
```

### Custom Judge Configuration

Configure domain-specific safety evaluation by customizing the judge model:

```python
# Aegis evaluation with custom judge configuration
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="/results/safety/aegis_custom/",
    params=ConfigParams(
        limit_samples=100,
        parallelism=1,
        temperature=0.7,
        extra={
            "judge": {
                "model_id": "your-custom-safety-judge",
                "url": "http://your-judge-endpoint:9000/v1/completions",
                "api_key": "your_judge_api_key",
                "parallelism": 8,
                "request_timeout": 60
            }
        }
    )
)
```

## Security Vulnerability Assessment

### Garak Security Scanning

Use Garak for comprehensive security vulnerability testing with configurable probes:

```python
# Security vulnerability evaluation with specific probes
eval_config = EvaluationConfig(
    type="garak",
    output_dir="/results/security/garak/",
    params=ConfigParams(
        limit_samples=50,
        temperature=0.8,      # Varied responses for security testing
        parallelism=2,
        extra={
            "probes": "promptinject,leakage.DivergenceInject,encoding.InjectAscii85"
        }
    )
)

security_result = evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config
)

# For all available probes, omit the probes parameter or set to None
eval_config_all = EvaluationConfig(
    type="garak",
    output_dir="/results/security/garak_all/",
    params=ConfigParams(
        limit_samples=50,
        extra={"probes": None}  # Runs all available probes
    )
)
```

## Understanding Safety Metrics

### Safety Evaluation Results

Safety evaluation results are returned in the standardized `EvaluationResult` format. The specific metrics vary by task:

**Aegis V2 Results**:
The `aegis_v2` task returns safety scores based on the NemoGuard judge model's assessment. Results are saved to the `results.yml` file in your output directory and follow the standard evaluation result structure with task-specific metrics.

**Garak Results**:
The `garak` task returns pass/fail rates for each probe executed, along with detailed vulnerability reports.

Refer to the generated `results.yml` and `report.html` files in your output directory for detailed metrics and interpretations specific to your evaluation.

### Interpreting Results

Safety evaluation results should be interpreted in the context of your specific use case and deployment environment. Consider:

- **Pass Rates**: Higher pass rates indicate better safety alignment
- **Vulnerability Detection**: Pay attention to any detected vulnerabilities or failures
- **Judge Model Assessments**: Review detailed judge model responses for context
- **Probe Coverage**: For Garak, review which probes were tested and their results

Refer to your organization's safety guidelines and thresholds when determining deployment readiness.

## Production Safety Monitoring

### Continuous Safety Assessment

For production monitoring, you can periodically run safety evaluations on sample production data:

```python
# Production safety evaluation
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="/results/production_monitoring/",
    params=ConfigParams(
        limit_samples=1000,    # Sample size for monitoring
        parallelism=4,
        temperature=0.7,
        extra={
            "judge": {
                "model_id": "llama-nemotron-safety-guard-v2",
                "url": "http://safety-judge:9000/v1/completions"
            }
        }
    )
)

# Run evaluation on production sample data
monitoring_result = evaluate(
    target_cfg=target_config,
    eval_cfg=eval_config
)
```

## Judge Model Requirements

### Deploying NemoGuard Safety Judge

```bash
# Example NemoGuard deployment (adjust for your environment)
docker run -d --name safety-judge \
  --gpus all \
  -p 9000:8000 \
  nvcr.io/nvidia/nemoguard:latest
```

For detailed NemoGuard setup, see the [NemoGuard documentation](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/).

---

*For comprehensive safety evaluation strategies, see the [Safety Harness documentation](https://pypi.org/project/nvidia-safety-harness/) and [Garak security scanner](https://pypi.org/project/nvidia-eval-factory-garak/).*
