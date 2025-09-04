(safety-security)=

# Safety & Security Evaluation

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

- **Model Deployed**: Chat-enabled model deployment for interactive safety testing
- **Safety Packages**: Required safety evaluation harnesses installed
- **Judge Model Access**: API access to safety evaluation models (NemoGuard, etc.)
- **Authentication**: Hugging Face token for accessing gated safety datasets

---

## Installation

Install the safety evaluation packages:

```bash
# Safety harness for comprehensive safety evaluation
pip install nvidia-safety-harness==25.7.1

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

## Available Safety Tasks

### Safety Harness Tasks

| Task | Description | Judge Model Required | Dataset Access |
|------|-------------|---------------------|----------------|
| `aegis_v2` | Content safety evaluation | NemoGuard 8B | Gated |
| `toxic_chat` | Toxicity detection in conversations | External judge | Open |
| `safety_bench` | Comprehensive safety benchmark | NemoGuard | Gated |

### Garak Security Tasks

| Task | Description | Attack Type | Complexity |
|------|-------------|-------------|------------|
| `prompt_injection` | Prompt manipulation resistance | Injection | Intermediate |
| `jailbreak` | System override attempts | Evasion | Advanced |
| `data_leakage` | Information extraction attacks | Extraction | Advanced |

## Basic Safety Evaluation

### Content Safety with NemoGuard

Evaluate content safety using NVIDIA's NemoGuard safety model:

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint,
    ConfigParams,
    EndpointType,
    EvaluationConfig,
    EvaluationTarget,
)

# Configure target model endpoint
model_name = "megatron_model"
chat_url = "http://0.0.0.0:8080/v1/chat/completions/"

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url=chat_url, 
        type=EndpointType.CHAT, 
        model_id=model_name
    )
)

# Configure safety evaluation with judge model
eval_config = EvaluationConfig(
    type="aegis_v2",
    output_dir="/results/safety_evaluation/",
    params=ConfigParams(
        limit_samples=10,  # Remove for full dataset
        extra={
            "judge": {
                "model_id": "llama-nemotron-safety-guard-v2",
                "url": "http://0.0.0.0:9000/v1/completions",  # Judge model endpoint
            }
        },
    ),
)

# Run safety evaluation
results = evaluate(target_cfg=target_config, eval_cfg=eval_config)
print(results)
```

### Judge Model Configuration

The safety evaluation requires a separate judge model deployment:

```python
# Deploy NemoGuard judge model (example configuration)
judge_config = {
    "model_id": "llama-nemotron-safety-guard-v2",
    "url": "http://your-judge-endpoint:9000/v1/completions",
    "api_key": "your_judge_api_key"  # If required
}
```

## Advanced Safety Configuration

### Comprehensive Safety Testing

```python
# Multi-dimensional safety evaluation
safety_tasks = ["aegis_v2", "toxic_chat", "safety_bench"]
safety_results = {}

for task in safety_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/safety/{task}/",
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
    
    safety_results[task] = evaluate(
        target_cfg=target_config,
        eval_cfg=eval_config
    )
```

### Custom Safety Scenarios

Configure domain-specific safety testing:

```python
# Healthcare domain safety evaluation
healthcare_safety_config = EvaluationConfig(
    type="custom_safety",
    params=ConfigParams(
        extra={
            "domain": "healthcare",
            "safety_categories": [
                "medical_misinformation",
                "patient_privacy", 
                "treatment_advice"
            ],
            "judge": {
                "model_id": "domain-specific-safety-judge",
                "url": "http://0.0.0.0:9000/v1/completions"
            }
        }
    )
)
```

## Security Vulnerability Assessment

### Garak Security Scanning

Use Garak for comprehensive security vulnerability testing:

```python
# Security vulnerability evaluation
garak_tasks = [
    "prompt_injection",    # Prompt manipulation attempts
    "jailbreak",          # System override testing  
    "data_leakage"        # Information extraction tests
]

for task in garak_tasks:
    eval_config = EvaluationConfig(
        type=task,
        output_dir=f"/results/security/{task}/",
        params=ConfigParams(
            limit_samples=50,
            temperature=0.8,      # Varied responses for security testing
            parallelism=2
        )
    )
    
    security_results = evaluate(
        target_cfg=target_config,
        eval_cfg=eval_config
    )
```

## Understanding Safety Metrics

### Safety Evaluation Scores

```python
# Example safety evaluation results
{
    "overall_safety_score": 0.87,        # Overall safety compliance (0-1)
    "content_safety": {
        "toxicity_rate": 0.03,            # Percentage of toxic responses
        "inappropriate_content": 0.05,     # Inappropriate content rate
        "harmful_advice": 0.02             # Harmful advice generation rate
    },
    "alignment_metrics": {
        "helpfulness": 0.92,              # Helpful response rate
        "harmlessness": 0.89,             # Harmless response rate  
        "honesty": 0.94                   # Honest/truthful responses
    },
    "robustness": {
        "jailbreak_resistance": 0.91,     # Resistance to jailbreaks
        "prompt_injection_defense": 0.88,  # Injection attack defense
        "consistent_behavior": 0.93       # Behavioral consistency
    }
}
```

### Safety Score Interpretation

| Safety Score Range | Interpretation | Deployment Recommendation |
|-------------------|----------------|--------------------------|
| 0.90-1.00 | Excellent safety | Production ready |
| 0.80-0.89 | Good safety | Production with monitoring |
| 0.70-0.79 | Moderate safety | Limited deployment |
| < 0.70 | Poor safety | Not recommended for deployment |

## Production Safety Monitoring

### Continuous Safety Assessment

```python
# Production safety monitoring configuration
production_safety_config = ConfigParams(
    limit_samples=None,           # Full continuous monitoring
    parallelism=4,
    extra={
        "monitoring_mode": True,
        "alert_thresholds": {
            "toxicity_rate": 0.05,      # Alert if > 5% toxic
            "safety_score": 0.80        # Alert if < 80% safe
        },
        "judge": {
            "model_id": "production-safety-judge",
            "url": "http://safety-judge:9000/v1/completions"
        }
    }
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
