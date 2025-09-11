(gs-quickstart-launcher)=
# NeMo Evaluator Launcher

**Best for**: Most users who want a unified CLI experience

The NeMo Evaluator Launcher provides the simplest way to run evaluations with automated container management, built-in orchestration, and comprehensive result export capabilities.

## Prerequisites

- OpenAI-compatible endpoint (hosted or self-deployed)
- Docker installed (for local execution)

## Quick Start

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

## Complete Working Example

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

## Key Features

### Automated Container Management

- Automatically pulls and manages evaluation containers
- Handles GPU access and volume mounting
- No manual Docker commands required

### Built-in Orchestration

- Job queuing and parallel execution
- Progress monitoring and status tracking
- Automatic retry on failures

### Result Export

- Export to MLflow, Weights & Biases, or local formats
- Structured result formatting
- Integration with experiment tracking platforms

### Configuration Management

- YAML-based configuration system
- Override parameters via command line
- Template configurations for common scenarios

## Next Steps

- Explore different evaluation types: `nv-eval ls tasks`
- Try advanced configurations in the `examples/` directory
- Export results to your preferred tracking platform
- Scale to cluster execution with Slurm or cloud providers

For more advanced control, consider the {ref}`gs-quickstart-core` Python API or {ref}`gs-quickstart-container` approaches.
