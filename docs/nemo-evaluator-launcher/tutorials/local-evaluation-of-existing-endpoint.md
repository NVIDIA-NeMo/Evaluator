# Local Evaluation of Existing Endpoint

This tutorial shows how to evaluate an existing API endpoint using the Local executor.

## Prerequisites

# Installation
First, install the NeMo Evaluator Launcher. See the [Quickstart Installation Guide](../quickstart.md#1-install-the-launcher) for detailed setup instructions.

# Requirements
- Docker
- Python environment with the Nemo Evaluator Launcher CLI available

## Step-by-Step Guide

# 1. Select Model

You have two options:

## Option A: Use NVIDIA Build API or another hosted endpoint
- **URL**: `https://integrate.api.nvidia.com/v1/chat/completions` (or your hosted endpoint)
- **Models**: You can select any OpenAI‑compatible endpoint, including those from the extensive catalog on NVIDIA Build
- **API Key**: Get from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct) (or your provider)
  - For NVIDIA APIs, see [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)

## Option B: Deploy Your Own Endpoint
Deploy an OpenAI-compatible endpoint using frameworks like vLLM, SGLang, NeMo, TRT-LLM, or NIM. See examples: [Deployment Frameworks Guide](deployments/deployment-frameworks-guide.md)

/// note | Tutorial Example
For this tutorial we will use `meta/llama-3.1-8b-instruct` from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct).
///

# 2. Select Tasks

Choose which benchmarks to evaluate. Available tasks include:

```bash
nemo-evaluator-launcher ls tasks
```

For a comprehensive list of supported tasks and descriptions, see the [NeMo Evaluator supported tasks](../../nemo-evaluator/reference/containers.md).

**Important**: Each task has a dedicated endpoint type (e.g., `/v1/chat/completions`, `/v1/completions`). Ensure that your model provides the correct endpoint type for the tasks you want to evaluate.

/// note | Tutorial Example
For this tutorial we will pick: `ifeval` and `humaneval_instruct` as these are relatively fast, both use the chat endpoint.
///


# 3. Create configuration file

Create a `configs` directory and your first configuration file:

```bash
mkdir configs
```

Create a configuration file with a descriptive name (e.g., `configs/local_endpoint.yaml`):

This configuration will create evaluations for 2 tasks: `ifeval` and `humaneval_instruct`. You can display the whole configuration and scripts which will be executed using `--dry_run`

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
    api_key_name: API_KEY  # API Key with access to build.nvidia.com or model of your choice

# specify the benchmarks to evaluate
evaluation:
  overrides:  # these overrides apply to all tasks; for task-specific overrides, use the `overrides` field
    config.params.request_timeout: 3600
  tasks:
    - name: ifeval  # use the default benchmark configuration
    - name: humaneval_instruct
```

# 4. Run evaluation

```bash
nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.api_key=API_KEY
```

# 5. Run  the same evaluation for a different model (using CLI overrides)

```bash
export API_KEY=<YOUR MODEL API KEY>
MODEL_NAME=<YOUR_MODEL_NAME>
URL=<YOUR_ENDPOINT_URL>  # Note: endpoint URL needs to be FULL (e.g., https://api.example.com/v1/chat/completions)

nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.model_id=$MODEL_NAME \
  -o target.api_endpoint.url=$URL \
  -o target.api_endpoint.api_key=API_KEY
```

After the launch you can monitor lively logs, status and after finishing display results and optionally export them in a unified nemo evaluator launcher way. After the failure e.g. connection error you can resume the job without the data loss [resuming] See [Exporters Documentation](../../exporters/overview.md) for available export options.

## Next Steps

- **[Advanced Task Configuration](../../configuration/evaluation/index.md)**: Customize evaluation parameters and prompts
- **[Different Executors](../../executors/overview.md)**: Try Slurm or Lepton for different environments
- **[Deploy Your Own Models](deployments/deployment_frameworks_guide.md)**: Use vLLM, SGLang, or NIM
- **[Test Endpoint Compatibility](deployments/testing-endpoint-oai-compatibility.md)**: Verify your endpoint with curl requests
- **[Export Results](../../exporters/overview.md)**: Send results to W&B, MLFlow, or other platforms 