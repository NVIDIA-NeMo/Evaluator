(tutorials-local-eval-existing-endpoint)=
# Local Evaluation of Existing Endpoint

This tutorial shows how to evaluate an existing API endpoint using the Local executor.

## Prerequisites

- Docker
- Python environment with the NeMo Evaluator Launcher CLI available (install the launcher by following {ref}`gs-install`)

## Step-by-Step Guide

### 1. Select a Model

You have the following options:

#### Option I: Use the NVIDIA Build API

- **URL**: `https://integrate.api.nvidia.com/v1/chat/completions`
- **Models**: Choose any endpoint from NVIDIA Build's extensive catalog
- **API Key**: Get from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct). See [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key).
  Make sure to export the API key:

```
export NGC_API_KEY=nvapi-...
```

#### Option II: Another Hosted Endpoint

- **URL**: Your model's endpoint URL
- **Models**: Any OpenAI-compatible endpoint
- **API_KEY**: If your endpoint is gated, get an API key from your provider and export it:

```
export API_KEY=...
```

#### Option III: Deploy Your Own Endpoint

Deploy an OpenAI-compatible endpoint using frameworks like vLLM, SGLang, TRT-LLM, or NIM.
<!-- TODO: uncomment ref once the guide is ready -->
<!-- Refer to {ref}`bring-your-own-endpoint-manual` for deployment guidance -->

:::{note}
For this tutorial, we will use `meta/llama-3.1-8b-instruct` from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct). You will need to export your `NGC_API_KEY` to access this endpoint.
:::

### 2. Select Tasks

Choose which benchmarks to evaluate. You can list all available tasks with the following command:

```bash
nemo-evaluator-launcher ls tasks
```

For a comprehensive list of supported tasks and descriptions, see {ref}`nemo-evaluator-containers`.

**Important**: Each task has a dedicated endpoint type (e.g., `/v1/chat/completions`, `/v1/completions`). Ensure that your model provides the correct endpoint type for the tasks you want to evaluate. Use our {ref}`deployment-testing-compatibility` guide to verify your endpoint supports the required formats.

:::{note}
For this tutorial we will pick: `ifeval` and `humaneval_instruct` as these are fast. They both use the chat endpoint.
:::

### 3. Create a Configuration File

Create a `configs` directory:

```bash
mkdir configs
```

Create a configuration file with a descriptive name (e.g., `configs/local_endpoint.yaml`)
and populate it with the following content:

```yaml
defaults:
  - execution: local  # The evaluation will run locally on your machine using Docker
  - deployment: none  # Since we are evaluating an existing endpoint,  we don't need to deploy the model
  - _self_

execution:
  output_dir: results/${target.api_endpoint.model_id}  # Logs and artifacts will be saved here
  mode: sequential # Default: run tasks sequentially. You can also use the mode 'parallel'

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct  # TODO: update to the model you want to evaluate
    url: https://integrate.api.nvidia.com/v1/chat/completions  # TODO: update to the endpoint you want to evaluate
    api_key_name: NGC_API_KEY  # Name of the env variable that stores the API Key with access to build.nvidia.com (or model of your choice)

# specify the benchmarks to evaluate
evaluation:
  # Optional: Global evaluation overrides - these apply to all benchmarks below
  nemo_evaluator_config:
    config:
      params:
        parallelism: 2
        request_timeout: 1600
  tasks:
    - name: ifeval  # use the default benchmark configuration
    - name: humaneval_instruct
      # Optional: Task overrides - here they apply only to the task `humaneval_instruct`
      nemo_evaluator_config:
        config:
          params:
            max_new_tokens: 1024
            temperature: 0.3
```

This configuration will create evaluations for 2 tasks: `ifeval` and `humaneval_instruct`.

You can display the whole configuration and scripts which will be executed using `--dry-run`:

```
nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint --dry-run
```

### 4. Run the Evaluation

Once your configuration file is complete, you can run the evaluations:

```bash
nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint
```

### 5. Run the Same Evaluation for a Different Model (Using CLI Overrides)
You can override the values from your configuration file using CLI overrides:

```bash
export API_KEY=<YOUR MODEL API KEY>
MODEL_NAME=<YOUR_MODEL_NAME>
URL=<YOUR_ENDPOINT_URL>  # Note: endpoint URL needs to be FULL (e.g., https://api.example.com/v1/chat/completions)

nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.model_id=$MODEL_NAME \
  -o target.api_endpoint.url=$URL \
  -o target.api_endpoint.api_key_name=API_KEY
```

### 6. Check the Job Status and Results

List the runs from last 2 hours to see the invocation IDs of the two evaluation jobs:

```bash
nemo-evaluator-launcher ls runs --since 2h   # list runs from last 2 hours
```

Use the IDs to check the jobs statuses:

```bash
nemo-evaluator-launcher status <invocation_id1> <invocation_id2> --json
```

When jobs finish, you can display results and export them using the available exporters:

```bash
# Check the results
cat results/*/artifacts/results.yml

# Check the running logs
tail -f results/*/*/logs/stdout.log   # use the output_dir printed by the run command

# Export metrics and metadata from both runs to json
nemo-evaluator-launcher export <invocation_id1> <invocation_id2> --dest local --format json
cat processed_results.json
```

Refer to {ref}`exporters-overview` for available export options.

## Next Steps

- **{ref}`evaluation-configuration`**: Customize evaluation parameters and prompts
- **{ref}`executors-overview`**: Try Slurm or Lepton for different environments
<!-- TODO: uncoment once ready -->
<!-- - **{ref}`bring-your-own-endpoint-manual`**: Deploy your own endpoints with various frameworks -->
- **{ref}`exporters-overview`**: Send results to W&B, MLFlow, or other platforms
