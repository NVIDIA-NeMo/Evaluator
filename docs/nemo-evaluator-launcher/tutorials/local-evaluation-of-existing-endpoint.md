# Local Evaluation of Existing Endpoint

This tutorial shows how to evaluate an existing API endpoint using the Local executor.

## Prerequisites

### Installation

First, install the NeMo Evaluator Launcher. Refer to the [Tutorial Installation Guide](../tutorial.md#1-install-the-launcher) for detailed setup instructions.

### Requirements

- Docker
- Python environment with the NeMo Evaluator Launcher CLI available

## Step-by-Step Guide

### 1. Select Model

You have the following two options:

## Option A: Use NVIDIA Build API or another hosted endpoint

- **URL**: `https://integrate.api.nvidia.com/v1/chat/completions` (or your hosted endpoint)
- **Models**: You can select any OpenAI‑compatible endpoint, including those from the extensive catalog on NVIDIA Build
- **API Key**: Get from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct) (or your provider)
  - For NVIDIA APIs, refer to [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)

## Option B: Deploy Your Own Endpoint

Deploy an OpenAI-compatible endpoint using frameworks like vLLM, SGLang, NeMo, TRT-LLM, or NIM. Refer to examples: [Deployment Frameworks Guide](deployments/deployment-frameworks-guide.md)

/// note | Tutorial Example
For this tutorial we will use `meta/llama-3.1-8b-instruct` from [build.nvidia.com](https://build.nvidia.com/meta/llama-3_1-8b-instruct).
///

### 2. Select Tasks

Choose which benchmarks to evaluate. Available tasks include the following:

```bash
nemo-evaluator-launcher ls tasks
```

For a comprehensive list of supported tasks and descriptions, refer to the [NeMo Evaluator supported tasks](../../nemo-evaluator/reference/containers.md).

**Important**: Each task has a dedicated endpoint type (for example, `/v1/chat/completions`, `/v1/completions`). Ensure that your model provides the correct endpoint type for the tasks you want to evaluate.

/// note | Tutorial Example
For this tutorial we will pick: `ifeval` and `humaneval_instruct` as these are relatively fast, both use the chat endpoint.
///


### 3. Create Configuration File

Create a `configs` directory and your first configuration file:

```bash
mkdir configs
```

Create a configuration file with a descriptive name (e.g., `configs/local_endpoint.yaml`):

This configuration will create evaluations for two tasks: `ifeval` and `mbpp`.

- **`ifeval`**: [Instruction Following Evaluation](https://arxiv.org/abs/2311.07911) - evaluates ability to follow natural language instructions
- **`mbpp`**: [Mostly Basic Programming Problems](https://arxiv.org/abs/2108.07732) - coding benchmark with 974 Python programming tasks

**Expected runtime**: For the given parallelism `ifeval` takes approximately 25 minutes, `mbpp` takes approximately one hour.

**Quick test**: For a faster sanity check, uncomment `config.params.limit_samples: 10` in the configuration below to run only 10 samples per task.

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
  overrides: # Note: The overrides above apply to all tasks in the evaluation
    config.params.request_timeout: 3600  # Maximum time (in seconds) to wait for each API response
    config.params.parallelism: 4         # Number of parallel requests to send simultaneously
    # config.params.limit_samples: 10 # **SANITY CHECK**: Uncomment this line to run only 10 samples per task for quick testing
  tasks:
    - name: ifeval  # use the default benchmark configuration
    - name: mbpp

```

You can display the whole configuration and scripts that will be executed using `--dry_run`:

### 4. Run Evaluation

```bash
nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.api_key_name=API_KEY
```

### 5. Run the Same Evaluation for a Different Model (Using CLI Overrides)

```bash
export API_KEY=<YOUR MODEL API KEY>
MODEL_NAME=<YOUR_MODEL_NAME>
URL=<YOUR_ENDPOINT_URL>  # Note: endpoint URL needs to be FULL (e.g., https://api.example.com/v1/chat/completions)

nemo-evaluator-launcher run --config-dir configs --config-name local_endpoint \
  -o target.api_endpoint.model_id=$MODEL_NAME \
  -o target.api_endpoint.url=$URL \
  -o target.api_endpoint.api_key_name=API_KEY
```

After the successful launch you will see:

```text
Commands for real-time monitoring:
  tail -f /path/to/your/results/meta/llama-3.1-8b-instruct/20250924_102224-7afae8cd/ifeval/logs/stdout.log
  tail -f /path/to/your/results/meta/llama-3.1-8b-instruct/20250924_102224-7afae8cd/mbpp/logs/stdout.log

Follow all logs for this invocation:
  tail -f /path/to/your/results/meta/llama-3.1-8b-instruct/20250924_102224-7afae8cd/*/logs/stdout.log
Complete run config saved to: /home/user/.nemo-evaluator/run_configs/7afae8cd_config.yml
to check status: nemo-evaluator-launcher status 7afae8cd
to kill all jobs: nemo-evaluator-launcher kill 7afae8cd
to kill individual jobs: nemo-evaluator-launcher kill <job_id> (for example, 7afae8cd.0)
```

You can monitor your job status:

```text
nemo-evaluator-launcher status <JOB ID>
```

| Job ID     | Status  | Container                     | Location                                    |
|------------|---------|-------------------------------|---------------------------------------------|
| 7afae8cd.0 | running | ifeval-20250924_102224_113553 | <output_dir>/20250924_102224-7afae8cd/ifeval |
| 7afae8cd.1 | running | mbpp-20250924_102224_114140   | <output_dir>/20250924_102224-7afae8cd/mbpp   |

Both tasks (`ifeval` and `mbpp`) are launched simultaneously in parallel mode by default, allowing for faster evaluation completion as the tasks run concurrently rather than sequentially.

You can monitor live logs, status and after finishing display results and optionally export them in a unified NeMo Evaluator Launcher way. After a failure such as connection error, you can resume the job without data loss [resuming] Refer to [Exporters Documentation](nemo-evaluator-launcher/exporters/overview.md) for available export options.

## Results 

After evaluation completion, your output directory will contain the following structure:

```text
<output_dir>/20250924_102224-7afae8cd/
├── ifeval/
│   ├── artifacts/
│   │   ├── run_config.yml               # Task-specific configuration
│   │   ├── eval_factory_metrics.json    # Evaluation metrics and statistics
│   │   ├── results.yml                  # Detailed results in YAML format
│   │   ├── report.html                  # Human-readable HTML report
│   │   ├── report.json                  # JSON format report
│   │   └── <task specific artifacts>    # Task-specific artifacts
│   ├── logs/
│   │   ├── stdout.log                   # Main execution logs
│   │   ├── stage.pre-start              # Pre-start stage logs
│   │   └── stage.running                # Runtime stage logs
│   └── run.sh                           # Execution script
├── mbpp/
│   ├── artifacts/
│   │   ├── run_config.yml               # Task-specific configuration
│   │   ├── eval_factory_metrics.json    # Evaluation metrics and statistics
│   │   ├── results.yml                  # Detailed results in YAML format
│   │   ├── report.html                  # Human-readable HTML report
│   │   ├── report.json                  # JSON format report
│   │   └── <task specific artifacts>    # Task-specific artifacts
│   ├── logs/
│   │   ├── stdout.log                   # Main execution logs
│   │   ├── stage.pre-start              # Pre-start stage logs
│   │   └── stage.running                # Runtime stage logs
│   └── run.sh                           # Execution script
└── run_all.sequential.sh                # Script to run all tasks sequentially
```

## Key Artifacts

### NeMo Evaluator Artifacts

- **`results.yml`**: [Comprehensive results in YAML format, including scores and detailed metrics](../../nemo-evaluator/reference/outputs.md#resultsyml)
- **`run_config.yml`**: [NeMo-evaluator task-specific configuration used during execution](../../nemo-evaluator/reference/outputs.md#run-configyml)
- **`eval_factory_metrics.json`**: [Contains evaluation metrics, timing statistics, and performance data](../../nemo-evaluator/reference/outputs.md#eval-factory-metricsjson)
- **`report.html`**: [Human-readable report with example request-response pairs](../../nemo-evaluator/reference/outputs.md#reporthtml)

For detailed information about all output artifacts, refer to the [Output Reference](../../nemo-evaluator/reference/outputs.md).

### NeMo Evaluator Launcher Artifacts

- **`logs/`**: Execution logs including stdout.log and stage logs
- **`run.sh`**: Execution script for running the evaluation task

To get the final evaluation metrics, check `results.yml`

## Next Steps

- **[Advanced Task Configuration](nemo-evaluator-launcher/configuration/evaluation/index.md)**: Customize evaluation parameters and prompts
- **[Different Executors](nemo-evaluator-launcher/executors/overview.md)**: Try Slurm or Lepton for different environments
- **[Deploy Your Own Models](deployments/deployment_frameworks_guide.md)**: Use vLLM, SGLang, or NIM
- **[Test Endpoint Compatibility](deployments/testing-endpoint-oai-compatibility.md)**: Verify your endpoint with curl requests
- **[Export Results](nemo-evaluator-launcher/exporters/overview.md)**: Send results to W&B, MLFlow, or other platforms