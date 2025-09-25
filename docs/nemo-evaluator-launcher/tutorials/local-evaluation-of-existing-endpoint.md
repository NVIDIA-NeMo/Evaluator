# Local Evaluation of an Existing Endpoint

This tutorial explains how to benchmark an existing API endpoint using the local executor.

## Prerequisites

- Docker
- Python environment with the NeMo Evaluator Launcher CLI available

## Install the Launcher

First, install the NeMo Evaluator Launcher. Refer to the [Tutorial Installation Guide](../tutorial.md#1-install-the-launcher) for detailed setup instructions.

## Set Up Your Evaluation

## 1. Select a Model

You have two options:

### Option A: Use a hosted endpoint (for example, build.nvidia.com)

- **URL**: `https://integrate.api.nvidia.com/v1/chat/completions` (or your hosted endpoint)
- **Models**: You can select any OpenAI-compatible endpoint, including those from the catalog on build.nvidia.com
- **API key**: Get from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct) (or your provider)
  - For NVIDIA API keys, refer to [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)

### Option B: Deploy your own endpoint

Deploy an OpenAI-compatible endpoint using frameworks such as `vLLM`, `SGLang`, `NeMo`, `TensorRT-LLM`, or `NIM`. Refer to the [Deployment Frameworks Guide](deployments/deployment-frameworks-guide.md).

/// note | Tutorial Example
For this tutorial, we use `meta/llama-3.1-8b-instruct` from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct).
///

## 2. Select Tasks

Choose which benchmarks to run. Available tasks include:

```bash
nemo-evaluator-launcher ls tasks
```

For a comprehensive list of supported tasks and descriptions, refer to the [NeMo Evaluator supported tasks](../../nemo-evaluator/reference/containers.md).

**Important**: Each task has a dedicated endpoint type (for example, `/v1/chat/completions`, `/v1/completions`). Ensure that your model provides the correct endpoint type for the tasks that you plan to run.

/// note | Tutorial Example
For this tutorial, we select `ifeval` and `humaneval_instruct`. These tasks are fast and both use the chat endpoint.
///

## 3. Create the Configuration File

Create a `configs` directory and your first configuration file:

```bash
mkdir configs
```

Create a configuration file with a descriptive name (for example, `configs/local_endpoint.yaml`).

This configuration creates evaluations for two tasks: `ifeval` and `humaneval_instruct`. You can display the entire configuration and the scripts that will run by using `--dry_run`.

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: results/${target.api_endpoint.model_id}

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct  # TODO: update to the model you want to evaluate
    url: https://integrate.api.nvidia.com/v1/chat/completions  # TODO: update to the endpoint you want to evaluate
    api_key_name: API_KEY  # API key with access to build.nvidia.com or the model of your choice

# Specify the benchmarks to evaluate
evaluation:
  overrides:  # These overrides apply to all tasks; for task-specific overrides, use the `overrides` field
    config.params.request_timeout: 3600
  tasks:
    - name: ifeval  # Use the default benchmark configuration
    - name: humaneval_instruct
```

## 4. Run the Evaluation

```bash
nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.api_key=API_KEY
```

## 5. Reuse the Same Evaluation for a Different Model (CLI overrides)

```bash
export API_KEY=<YOUR MODEL API KEY>
MODEL_NAME=<YOUR_MODEL_NAME>
URL=<YOUR_ENDPOINT_URL>  # Endpoint URL must be the full path (for example, https://api.example.com/v1/chat/completions)

nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.model_id=$MODEL_NAME \
  -o target.api_endpoint.url=$URL \
  -o target.api_endpoint.api_key=API_KEY
```

After launch, you can view logs and status. When the evaluation finishes, display results and optionally export them using the NeMo Evaluator Launcher. If a job fails (for example, due to a connection error), you can resume the job without data loss. Refer to the [Exporters Overview](../exporters/index.md) for available export options.

## Next Steps

- **[Advanced Task Configuration](../configuration/evaluation/index.md)**: Customize evaluation parameters and prompts
- **[Executors](../executors/index.md)**: Try `Slurm` or `Lepton` for different environments
- **[Deploy Your Own Models](deployments/deployment-frameworks-guide.md)**: Use `vLLM`, `SGLang`, or `NIM`
- **[Test Endpoint Compatibility](deployments/testing-endpoint-oai-compatibility.md)**: Verify your endpoint with curl requests
- **[Export Results](../exporters/index.md)**: Send results to Weights & Biases (W&B), `MLflow`, or other platforms
