# Integration Patterns

This guide demonstrates how to combine the three tiers of NeMo Eval for complete evaluation workflows, from model deployment to result analysis.

## Overview

NeMo Eval's three-tier architecture provides flexible integration options:

- **Tier 1** (`nemo_eval`): Model deployment and serving
- **Tier 2** (`nemo-evaluator`): Advanced evaluation with adapters
- **Tier 3** (`nv-eval`): Workflow orchestration

## Pattern 1: Full Stack Integration

### Complete Workflow Example

```python
import os
from nemo_eval.api import deploy
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget
)

# Step 1: Deploy your model
deploy(
    nemo_checkpoint="/path/to/your/checkpoint.nemo",
    serving_backend="pytriton",
    server_port=8080,
    num_gpus=2,
    tensor_parallelism_size=2,
    max_input_len=4096,
    max_batch_size=8
)

# Step 2: Configure advanced evaluation with adapters
adapter_config = AdapterConfig(
    api_url="http://0.0.0.0:8080/v1/completions/",
    use_reasoning=True,
    end_reasoning_token="</think>",
    start_reasoning_token="<think>",
    custom_system_prompt="You are a helpful assistant that thinks step by step.",
    use_caching=True,
    caching_dir="./cache",
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=100,
    max_logged_responses=100
)

# Step 3: Configure evaluation target and benchmarks
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model"
)

target = EvaluationTarget(api_endpoint=api_endpoint)

config = EvaluationConfig(
    type="gsm8k",
    output_dir="results",
    params={"limit_samples": 50}
)

# Step 4: Run evaluation with full adapter pipeline
results = evaluate(
    target_cfg=target,
    eval_cfg=config,
    adapter_cfg=adapter_config
)

print(f"Evaluation completed. Results: {results}")
```

### What This Achieves

- **Model Deployment**: High-performance serving with tensor parallelism
- **Advanced Processing**: Chain-of-thought reasoning extraction
- **Caching**: Response caching for efficiency
- **Logging**: Comprehensive request/response logging
- **Custom Prompts**: Tailored system messages for specific tasks

## Pattern 2: Orchestrated Evaluation

### Using the Launcher for Production Workflows

```bash
# Create configuration file
cat > my_evaluation.yaml << EOF
defaults:
  - _self_

target:
  api_endpoint:
    url: http://0.0.0.0:8080/v1/completions/
    model_id: megatron_model
    adapter_config:
      use_reasoning: true
      end_reasoning_token: "</think>"
      start_reasoning_token: "<think>"
      custom_system_prompt: "You are a helpful assistant that thinks step by step."
      use_caching: true
      caching_dir: ./cache
      use_request_logging: true
      use_response_logging: true

config:
  type: gsm8k
  output_dir: results
  params:
    limit_samples: 100

execution:
  executor: local
  output_dir: ./evaluation_results
EOF

# Run orchestrated evaluation
nv-eval run --config-name my_evaluation
```

### Advanced Orchestration Features

```bash
# Multi-benchmark evaluation
nv-eval run \
  --config-name my_evaluation \
  -o config.type=gsm8k,hellaswag,arc_easy \
  -o execution.parallel_jobs=3

# Export results to multiple destinations
nv-eval export <invocation_id> \
  --dest mlflow,wandb,local \
  --format json

# Monitor progress
nv-eval status <invocation_id>
```

## Pattern 3: Custom Pipeline Integration

### Integrating into Existing ML Workflows

```python
class ModelEvaluationPipeline:
    def __init__(self, model_configs):
        self.model_configs = model_configs
        self.results = {}
    
    def evaluate_model(self, model_config):
        """Evaluate a single model configuration"""
        # Deploy model
        deploy(
            nemo_checkpoint=model_config['checkpoint'],
            serving_backend=model_config['backend'],
            server_port=model_config['port'],
            **model_config['deploy_args']
        )
        
        # Configure adapters based on model type
        adapter_config = self._create_adapter_config(model_config)
        
        # Run evaluations
        results = {}
        for benchmark in model_config['benchmarks']:
            eval_config = EvaluationConfig(
                type=benchmark,
                output_dir=f"results/{model_config['name']}/{benchmark}"
            )
            
            result = evaluate(
                target_cfg=self._create_target(model_config),
                eval_cfg=eval_config,
                adapter_cfg=adapter_config
            )
            results[benchmark] = result
        
        return results
    
    def _create_adapter_config(self, model_config):
        """Create adapter configuration based on model capabilities"""
        config = AdapterConfig(
            api_url=f"http://0.0.0.0:{model_config['port']}/v1/completions/",
            use_caching=True,
            caching_dir=f"./cache/{model_config['name']}"
        )
        
        # Enable reasoning for capable models
        if model_config.get('supports_reasoning', False):
            config.use_reasoning = True
            config.end_reasoning_token = "</think>"
            config.start_reasoning_token = "<think>"
        
        return config
    
    def run_comparative_study(self):
        """Run comparative evaluation across all models"""
        for model_config in self.model_configs:
            print(f"Evaluating {model_config['name']}...")
            self.results[model_config['name']] = self.evaluate_model(model_config)
        
        return self.generate_comparison_report()

# Usage
pipeline = ModelEvaluationPipeline([
    {
        'name': 'llama-3.1-8b',
        'checkpoint': '/path/to/llama-3.1-8b.nemo',
        'backend': 'pytriton',
        'port': 8080,
        'benchmarks': ['gsm8k', 'hellaswag', 'arc_easy'],
        'supports_reasoning': True,
        'deploy_args': {'num_gpus': 1}
    },
    {
        'name': 'mixtral-8x7b',
        'checkpoint': '/path/to/mixtral-8x7b.nemo',
        'backend': 'ray',
        'port': 8081,
        'benchmarks': ['gsm8k', 'hellaswag', 'arc_easy'],
        'supports_reasoning': False,
        'deploy_args': {'num_gpus': 4, 'num_replicas': 2}
    }
])

results = pipeline.run_comparative_study()
```

## Pattern 4: Distributed Evaluation

### Multi-Node Evaluation with Slurm

```yaml
# slurm_evaluation.yaml
defaults:
  - _self_

target:
  api_endpoint:
    url: http://node-1:8080/v1/completions/
    model_id: megatron_model
    adapter_config:
      use_caching: true
      caching_dir: /shared/cache
      use_request_logging: true
      use_response_logging: true

config:
  type: gsm8k,hellaswag,arc_easy,mmlu
  output_dir: /shared/results
  params:
    limit_samples: 1000

execution:
  executor: slurm
  slurm:
    nodes: 4
    gpus_per_node: 8
    time_limit: "04:00:00"
    partition: gpu
    job_name: nemo_eval_study
  parallel_jobs: 4
  output_dir: /shared/evaluation_results

deployment:
  type: nemo
  checkpoint: /shared/models/large-model.nemo
  backend: pytriton
  num_gpus: 8
  tensor_parallelism_size: 8
```

```bash
# Submit distributed evaluation
nv-eval run \
  --config-name slurm_evaluation \
  --executor slurm
```

## Pattern 5: Cloud-Native Evaluation

### Using Lepton AI for Scalable Evaluation

```yaml
# cloud_evaluation.yaml
defaults:
  - _self_

target:
  api_endpoint:
    url: https://your-model.lepton.run/v1/completions
    api_key: ${oc.env:LEPTON_API_KEY}
    model_id: your-model
    adapter_config:
      use_reasoning: true
      use_caching: true
      caching_dir: ./cloud_cache
      use_progress_tracking: true

config:
  type: gsm8k,hellaswag,arc_easy,mmlu,humaneval
  output_dir: cloud_results
  params:
    limit_samples: 500

execution:
  executor: lepton
  lepton:
    resource_shape: gpu.a100-80gb
    replicas: 4
    auto_scale: true
    max_replicas: 10
  parallel_jobs: 5
  output_dir: ./cloud_evaluation_results

exporters:
  - type: wandb
    config:
      project: model-evaluation
      tags: [cloud, production]
  - type: mlflow
    config:
      experiment_name: cloud-evaluation
```

## Best Practices

### 1. Resource Management

```python
# Use context managers for proper cleanup
from contextlib import contextmanager

@contextmanager
def evaluation_context(model_config):
    """Ensure proper resource cleanup"""
    try:
        # Deploy model
        deploy(**model_config)
        yield
    finally:
        # Cleanup resources (implementation specific)
        cleanup_deployment(model_config)

# Usage
with evaluation_context(model_config):
    results = evaluate(target_cfg, eval_cfg, adapter_cfg)
```

### 2. Error Handling and Retry Logic

```python
import time
from typing import Optional

def robust_evaluate(
    target_cfg, eval_cfg, adapter_cfg,
    max_retries: int = 3,
    retry_delay: float = 30.0
) -> Optional[dict]:
    """Evaluate with retry logic for transient failures"""
    
    for attempt in range(max_retries):
        try:
            return evaluate(target_cfg, eval_cfg, adapter_cfg)
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Attempt {attempt + 1} failed: {e}. Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"All {max_retries} attempts failed. Last error: {e}")
                raise
    
    return None
```

### 3. Configuration Management

```python
from omegaconf import OmegaConf

def load_evaluation_config(config_path: str, overrides: list = None):
    """Load and validate evaluation configuration"""
    config = OmegaConf.load(config_path)
    
    if overrides:
        for override in overrides:
            OmegaConf.update(config, override, merge=True)
    
    # Validate required fields
    required_fields = [
        'target.api_endpoint.url',
        'config.type',
        'config.output_dir'
    ]
    
    for field in required_fields:
        if not OmegaConf.select(config, field):
            raise ValueError(f"Required field '{field}' not found in configuration")
    
    return config
```

## Troubleshooting

### Common Integration Issues

1. **Port Conflicts**: Ensure deployment and adapter servers use different ports
2. **Resource Contention**: Monitor GPU memory usage during concurrent evaluations
3. **Network Timeouts**: Configure appropriate timeouts for large model responses
4. **Cache Management**: Implement cache cleanup for long-running evaluations

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use adapter logging for request/response inspection
adapter_config = AdapterConfig(
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=10,  # Limit for debugging
    log_failed_requests=True
)

# Monitor evaluation progress
adapter_config.use_progress_tracking = True
adapter_config.progress_tracking_url = "http://localhost:3828/progress"
```

This integration guide provides the foundation for building sophisticated evaluation workflows that leverage the full power of NeMo Eval's three-tier architecture.
