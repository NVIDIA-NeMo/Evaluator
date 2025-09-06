(gs-quickstart)=

# Quick Start

Get up and running with NeMo Evaluator in minutes. Choose your preferred approach based on your needs.

## Choose Your Path

### 🚀 **Path 1: NeMo Evaluator Launcher** (Recommended)

**Best for**: Most users who want a unified CLI experience

#### Prerequisites

- OpenAI-compatible endpoint (hosted or self-deployed)
- Docker installed (for local execution)

#### Quick Start

```bash
# 1. Install the launcher
pip install nemo-evaluator-launcher

# 2. List available benchmarks
nv-eval ls tasks

# 3. Run evaluation against a hosted endpoint
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY} \
    -o execution.output_dir=./results

# 4. Check status and results
nv-eval status <invocation_id>
```

#### Complete Working Example

Here's a complete example using NVIDIA Build (build.nvidia.com):

```bash
# Set up your API key
export NGC_API_KEY="nvapi-your-key-here"

# Run a quick test evaluation with limited samples
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.model_id=meta/llama-3.1-8b-instruct \
    -o target.api_endpoint.api_key=${NGC_API_KEY} \
    -o execution.output_dir=./results \
    -o +config.params.limit_samples=10

# Monitor progress (replace with actual invocation_id from output)
nv-eval status <invocation_id>

# View results
ls -la ./results/<invocation_id>/
```

**What happens:**

- Pulls appropriate evaluation container
- Runs benchmark against your endpoint
- Saves results to specified directory
- Provides monitoring and status updates

---

### ⚙️ **Path 2: NeMo Evaluator Core**

**Best for**: Developers who need programmatic control

#### Prerequisites

- Python environment with nemo-evaluator installed
- OpenAI-compatible endpoint

#### Quick Start

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, 
    EvaluationTarget, 
    ApiEndpoint, 
    EndpointType,
    ConfigParams,
    AdapterConfig
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,  # Small test run
        temperature=0.0,
        max_new_tokens=1024,
        parallelism=1
    )
)

# Configure target endpoint with adapter features
adapter_config = AdapterConfig(
    use_request_logging=True,      # Log all requests
    use_response_logging=True,     # Log all responses
    use_caching=True,              # Enable caching
    caching_dir="./cache"          # Cache directory
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        api_key="your_api_key_here",
        type=EndpointType.CHAT,
        adapter_config=adapter_config
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
```

#### Complete Working Example

Here's a comprehensive example with error handling and environment setup:

```python
import os
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, EvaluationTarget, ApiEndpoint, 
    EndpointType, ConfigParams, AdapterConfig
)

# Set up environment
os.environ["NGC_API_KEY"] = "nvapi-your-key-here"

# Configure evaluation with comprehensive settings
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=3,           # Small test for quick validation
        temperature=0.0,           # Deterministic results
        max_new_tokens=1024,
        parallelism=1,
        max_retries=5
    )
)

# Advanced adapter configuration
adapter_config = AdapterConfig(
    use_request_logging=True,
    use_response_logging=True,
    use_caching=True,
    caching_dir="./cache",
    save_responses=True,
    log_failed_requests=True
)

target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        model_id="meta/llama-3.1-8b-instruct",
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        type=EndpointType.CHAT,
        api_key=os.environ["NGC_API_KEY"],
        adapter_config=adapter_config
    )
)

# Run evaluation with error handling
try:
    print("Starting evaluation...")
    result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
    print(f"✅ Evaluation completed successfully!")
    print(f"Results saved to: {eval_config.output_dir}")
    print(f"Result summary: {result}")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")
    print("Check your API key and endpoint configuration")
```

---

### 🐳 **Path 3: Container Direct**

**Best for**: Users who prefer container-based workflows

#### Prerequisites

- Docker with GPU support
- OpenAI-compatible endpoint

#### Quick Start

```bash
# 1. Pull evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3

# 2. Run container interactively
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 bash

# 3. Inside container - set up environment
export MY_API_KEY=nvapi-your-key-here
export HF_TOKEN=hf_your-token-here  # If using Hugging Face models

# 4. Run evaluation
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results \
    --overrides 'config.params.limit_samples=10'
```

#### Complete Container Workflow

Here's a complete example with volume mounting and advanced configuration:

```bash
# 1. Create local directories for persistent storage
mkdir -p ./results ./cache ./logs

# 2. Run container with volume mounts
docker run --rm -it --gpus all \
    -v $(pwd)/results:/workspace/results \
    -v $(pwd)/cache:/workspace/cache \
    -v $(pwd)/logs:/workspace/logs \
    -e MY_API_KEY=nvapi-your-key-here \
    -e HF_TOKEN=hf_your-token-here \
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 bash

# 3. Inside container - run evaluation with advanced features
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /workspace/results \
    --overrides 'config.params.limit_samples=3,target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=/workspace/cache'

# 4. Exit container and check results
exit
ls -la ./results/
```

#### One-Liner Container Execution

For automated workflows, you can run everything in a single command:

```bash
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -e MY_API_KEY=nvapi-your-key-here \
    nvcr.io/nvidia/eval-factory/simple-evals:25.07.3 \
    eval-factory run_eval \
        --eval_type mmlu_pro \
        --model_id meta/llama-3.1-8b-instruct \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_type chat \
        --api_key_name MY_API_KEY \
        --output_dir /workspace/results \
        --overrides 'config.params.limit_samples=3'
```

---

### 🔧 **Path 4: Complete Stack Integration** (For Advanced Users)

**Best for**: Users who want complete control over deployment and evaluation

This path demonstrates the full three-tier architecture: deployment → evaluation → orchestration.

#### Prerequisites

- NeMo checkpoint or compatible model
- CUDA-compatible GPU(s)
- Python environment with all NeMo Eval components

#### Complete Workflow Example

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

print("🚀 Starting Complete Stack Integration Example")

# Step 1: Deploy Model (Tier 1)
print("\n📦 Step 1: Deploying model with nemo_eval...")
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
print("⏳ Waiting for model server to be ready...")
server_ready = wait_for_fastapi_server(
    base_url="http://0.0.0.0:8080",
    model_name="megatron_model",
    max_retries=60,
    retry_interval=5
)

if not server_ready:
    raise RuntimeError("Model server failed to start")

print("✅ Model server is ready!")

# Step 2: Configure Advanced Evaluation (Tier 2)
print("\n⚙️ Step 2: Configuring advanced evaluation with adapters...")

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

print(f"🧪 Step 3: Running evaluations on {len(benchmarks)} benchmarks...")

for benchmark in benchmarks:
    print(f"\n📊 Evaluating {benchmark}...")
    
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
print(f"\n📈 Step 4: Analysis complete!")
print("=" * 50)

for benchmark, result in results.items():
    if "error" not in result:
        print(f"✅ {benchmark}: {result}")
    else:
        print(f"❌ {benchmark}: {result['error']}")

print("\n🎯 Integration Features Demonstrated:")
print("  • Model deployment with optimized settings")
print("  • Chain-of-thought reasoning extraction")
print("  • Comprehensive request/response logging")
print("  • Response caching for efficiency")
print("  • Custom system prompts")
print("  • Multi-benchmark evaluation")
print("  • Error handling and progress tracking")

print(f"\n📁 Check results in: ./results/")
print(f"📁 Check cache in: ./evaluation_cache/")
print(f"📁 Check logs for detailed request/response data")
```

#### Alternative: Configuration-Based Full Stack

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

This demonstrates the power of NeMo Eval's three-tier architecture working together seamlessly.

---

## Model Endpoints

NeMo Evaluator works with any OpenAI-compatible endpoint. You have several options:

### **Hosted Endpoints** (Recommended)

- **NVIDIA Build**: [build.nvidia.com](https://build.nvidia.com) - Ready-to-use hosted models
- **OpenAI**: Standard OpenAI API endpoints  
- **Other providers**: Anthropic, Cohere, or any OpenAI-compatible API

### **Self-Hosted Options**

If you prefer to host your own models:

```bash
# vLLM (recommended for self-hosting)
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

# Or use other serving frameworks
# TRT-LLM, NeMo Framework, Ray Serve, etc.
```

See [Model Serving & Deployment](../deployment/index.md) for detailed deployment options.

---

## Next Steps

### Explore More Benchmarks

```bash
# List all available tasks
nv-eval ls tasks

# Run different evaluation types
nv-eval run --config-dir examples --config-name local_safety_evaluation
nv-eval run --config-dir examples --config-name local_code_generation
```

### Export Results

```bash
# Export to MLflow
nv-eval export <invocation_id> --dest mlflow

# Export to Weights & Biases  
nv-eval export <invocation_id> --dest wandb

# Export to local files
nv-eval export <invocation_id> --dest local --format json
```

### Scale to Clusters

```bash
# Run on Slurm cluster
nv-eval run --config-dir examples --config-name slurm_multi_gpu

# Run on Lepton AI
nv-eval run --config-dir examples --config-name lepton_deployment
```

## Validation and Troubleshooting

### Quick Validation Steps

Before running full evaluations, validate your setup:

```bash
# 1. Test your endpoint connectivity
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
    -H "Authorization: Bearer $NGC_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 10
    }'

# 2. Run a dry-run to validate configuration
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run

# 3. Run a minimal test with very few samples
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o +config.params.limit_samples=1 \
    -o execution.output_dir=./test_results
```

### Common Issues and Solutions

**API Key Issues:**
```bash
# Verify your API key is set correctly
echo $NGC_API_KEY

# Test with a simple curl request (see above)
```

**Container Issues:**
```bash
# Check Docker is running and has GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Pull the latest container if you have issues
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
```

**Configuration Issues:**
```bash
# Enable debug logging
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Check available evaluation types
nv-eval ls tasks
```

**Result Validation:**
```bash
# Check if results were generated
find ./results -name "*.json" -type f

# View summary results
cat ./results/<invocation_id>/summary.json
```

### Environment Setup Checklist

- [ ] API key configured and working
- [ ] Docker installed with GPU support (for container workflows)
- [ ] Python environment set up (for Python API)
- [ ] Network access to model endpoints
- [ ] Sufficient disk space for results and caching

## Getting Help

- **Documentation**: Browse the comprehensive guides in this documentation
- **Examples**: Check the `examples/` directory for configuration templates
- **Issues**: Report problems on [GitHub Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)

### Quick Reference

| Task | Command |
|------|---------|
| List benchmarks | `nv-eval ls tasks` |
| Run evaluation | `nv-eval run --config-dir examples --config-name <config>` |
| Check status | `nv-eval status <invocation_id>` |
| Export results | `nv-eval export <invocation_id> --dest local --format json` |
| Dry run | Add `--dry-run` to any run command |
| Test with few samples | Add `-o +config.params.limit_samples=3` |
