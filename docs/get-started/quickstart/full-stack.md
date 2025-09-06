(gs-quickstart-full-stack)=
# Complete Stack Integration

**Best for**: Users who want complete control over deployment and evaluation

This approach demonstrates the full three-tier architecture: deployment → evaluation → orchestration. It combines model deployment, advanced evaluation features, and orchestration for production workflows.

## Prerequisites

- NeMo checkpoint or compatible model
- CUDA-compatible GPU(s)
- Python environment with all NeMo Eval components

## Complete Workflow Example

```python
import os
import time
from pathlib import Path

# Tier 1: Model Deployment (nemo_eval)
from nemo_eval.api import deploy
from nemo_eval.utils.base import wait_for_fastapi_server

# Tier 2: Advanced Evaluation (nemo-evaluator) 
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams
)

print("Starting Complete Stack Integration Example")

# Step 1: Deploy Model (Tier 1)
print("\nStep 1: Deploying model with nemo_eval...")
deploy(
    nemo_checkpoint="/path/to/your/checkpoint.nemo",  # Replace with your checkpoint
    serving_backend="pytriton",
    server_port=8080,
    num_gpus=1,
    max_input_len=4096,
    max_batch_size=4,
    enable_cuda_graphs=True,
    enable_flash_decode=True
)

# Wait for deployment to be ready
print("Waiting for model server to be ready...")
server_ready = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    model_name="megatron_model",
    max_retries=60,
    retry_interval=5
)

if not server_ready:
    raise RuntimeError("Model server failed to start")

print("Model server is ready!")

# Step 2: Configure Advanced Evaluation (Tier 2)
print("\nStep 2: Configuring advanced evaluation with adapters...")

# Create sophisticated adapter configuration
adapter_config = AdapterConfig(
    api_url="http://0.0.0.0:8080/v1/completions/",
    
    # Enable reasoning extraction
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    
    # Enable comprehensive logging
    use_request_logging=True,
    use_response_logging=True,
    max_logged_requests=50,
    max_logged_responses=50,
    log_failed_requests=True,
    
    # Enable caching for efficiency
    use_caching=True,
    caching_dir="./evaluation_cache",
    reuse_cached_responses=True,
    
    # Custom system prompt
    custom_system_prompt="You are a helpful AI assistant. Think step by step before answering.",
    
    # Progress tracking
    use_progress_tracking=True,
    progress_tracking_url="http://localhost:3828/progress"
)

# Configure evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model",
    adapter_config=adapter_config
)

target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure multiple benchmarks
benchmarks = ["gsm8k", "hellaswag", "arc_easy"]
results = {}

print(f"Step 3: Running evaluations on {len(benchmarks)} benchmarks...")

for benchmark in benchmarks:
    print(f"\n🔄 Evaluating {benchmark}...")
    
    config = EvaluationConfig(
        type=benchmark,
        output_dir=f"./results/{benchmark}",
        params=ConfigParams(
            limit_samples=10,  # Small test run
            temperature=0.0,
            max_new_tokens=512,
            parallelism=1
        )
    )
    
    # Run evaluation with full adapter pipeline
    try:
        result = evaluate(
            target_cfg=target,
            eval_cfg=config,
            adapter_cfg=adapter_config
        )
        results[benchmark] = result
        print(f"✅ {benchmark} completed successfully")
    except Exception as e:
        print(f"❌ {benchmark} failed: {e}")
        results[benchmark] = {"error": str(e)}

# Step 4: Results Analysis
print(f"\n📊 Step 4: Analysis complete!")
print("=" * 50)

for benchmark, result in results.items():
    if "error" not in result:
        print(f"✅ {benchmark}: {result}")
    else:
        print(f"❌ {benchmark}: {result['error']}")

print("\n🚀 Integration Features Demonstrated:")
print("  • Model deployment with optimized settings")
print("  • Chain-of-thought reasoning extraction")
print("  • Comprehensive request/response logging")
print("  • Response caching for efficiency")
print("  • Custom system prompts")
print("  • Multi-benchmark evaluation")
print("  • Error handling and progress tracking")

print(f"\n📁 Check results in: ./results/")
print(f"💾 Check cache in: ./evaluation_cache/")
print(f"📝 Check logs for detailed request/response data")
```

## Alternative: Configuration-Based Full Stack

For production workflows, use YAML configuration:

```yaml
# full_stack_config.yaml
defaults:
  - _self_

# Tier 1: Deployment Configuration
deployment:
  type: nemo
  checkpoint: /path/to/your/checkpoint.nemo
  backend: pytriton
  server_port: 8080
  num_gpus: 1
  max_input_len: 4096
  max_batch_size: 4
  enable_cuda_graphs: true
  enable_flash_decode: true

# Tier 2: Advanced Evaluation with Adapters
target:
  api_endpoint:
    url: http://0.0.0.0:8080/v1/completions/
    model_id: megatron_model
    adapter_config:
      # Reasoning capabilities
      use_reasoning: true
      start_reasoning_token: "<think>"
      end_reasoning_token: "</think>"
      
      # Comprehensive logging
      use_request_logging: true
      use_response_logging: true
      max_logged_requests: 100
      max_logged_responses: 100
      log_failed_requests: true
      
      # Caching for efficiency
      use_caching: true
      caching_dir: ./evaluation_cache
      reuse_cached_responses: true
      
      # Custom prompting
      custom_system_prompt: "You are a helpful AI assistant. Think step by step."
      
      # Progress tracking
      use_progress_tracking: true

# Multi-benchmark evaluation
config:
  type: gsm8k,hellaswag,arc_easy,mmlu
  output_dir: ./results
  params:
    limit_samples: 25
    temperature: 0.0
    max_new_tokens: 512
    parallelism: 2

# Tier 3: Orchestration
execution:
  executor: local
  parallel_jobs: 2
  output_dir: ./full_stack_results
  
exporters:
  - type: local
    config:
      format: json
      include_raw_responses: true
  - type: mlflow
    config:
      experiment_name: full-stack-evaluation
```

```bash
# Run the complete stack with single command
nv-eval run --config-name full_stack_config
```

## Key Features

### Three-Tier Architecture

- **Tier 1**: Model deployment and serving
- **Tier 2**: Advanced evaluation with adapters
- **Tier 3**: Orchestration and result management

### Advanced Adapter Capabilities

- Chain-of-thought reasoning extraction
- Comprehensive request/response logging
- Response caching for efficiency
- Custom system prompts
- Progress tracking integration

### Production Features

- Multi-benchmark evaluation
- Error handling and recovery
- Resource optimization
- Scalable orchestration
- Result export integration

### Deployment Optimizations

- CUDA graph acceleration
- Flash attention support
- Batch processing optimization
- GPU memory management

## Architecture Benefits

This demonstrates the power of NeMo Evaluator's three-tier architecture working together seamlessly:

1. **Model Deployment**: Optimized serving with hardware acceleration
2. **Advanced Evaluation**: Rich adapter features and comprehensive logging
3. **Orchestration**: Job management, result export, and scaling

## Use Cases

### Research and Development
- Model comparison across various benchmarks
- Ablation studies with different configurations
- Performance analysis with detailed logging

### Production Deployment
- Automated evaluation pipelines
- Continuous model validation
- Integration with machine learning operations workflows

### Enterprise Integration
- Custom adapter development
- Advanced logging and monitoring
- Integration with existing infrastructure

## Next Steps

- Adapt the configuration for your specific model checkpoints
- Explore custom adapter development
- Integrate with your machine learning operations pipeline
- Scale to cluster deployment
- Consider simpler approaches like [NeMo Evaluator Launcher](launcher.md) for standard workflows
