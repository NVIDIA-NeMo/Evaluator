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
nemo-evaluator-launcher ls tasks

# 3. Run evaluation against a hosted endpoint
nemo-evaluator-launcher run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${YOUR_API_KEY} \
    -o execution.output_dir=./results

# 4. Check status and results
nemo-evaluator-launcher status <invocation_id>
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
    ConfigParams
)

# Configure evaluation
eval_config = EvaluationConfig(
    type="mmlu_pro",
    output_dir="./results",
    params=ConfigParams(
        limit_samples=10,  # Small test run
        temperature=0.0
    )
)

# Configure target endpoint
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        api_key="your_api_key_here",
        type="chat"
    )
)

# Run evaluation
result = evaluate(eval_cfg=eval_config, target_cfg=target_config)
print(f"Evaluation completed: {result}")
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
export MY_API_KEY=your_api_key_here

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
nemo-evaluator-launcher ls tasks

# Run different evaluation types
nemo-evaluator-launcher run --config-dir examples --config-name local_safety_evaluation
nemo-evaluator-launcher run --config-dir examples --config-name local_code_generation
```

### Export Results
```bash
# Export to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases  
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to local files
nemo-evaluator-launcher export <invocation_id> --dest local --format json
```

### Scale to Clusters
```bash
# Run on Slurm cluster
nemo-evaluator-launcher run --config-dir examples --config-name slurm_multi_gpu

# Run on Lepton AI
nemo-evaluator-launcher run --config-dir examples --config-name lepton_deployment
```

## Getting Help

- **Documentation**: Browse the comprehensive guides in this documentation
- **Examples**: Check the `examples/` directory for configuration templates
- **Issues**: Report problems on [GitHub Issues](https://github.com/NVIDIA-NeMo/Eval/issues)
- **Community**: Join discussions on [GitHub Discussions](https://github.com/NVIDIA-NeMo/Eval/discussions)
