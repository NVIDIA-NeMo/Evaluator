# Local Executor

The Local executor runs evaluations on your machine using Docker. It provides a fast way to iterate if you have Docker installed, evaluating existing endpoints.

See common concepts and commands in the [executors overview](overview.md).

## Prerequisites

- Docker
- Python environment with the Nemo Evaluator Launcher CLI available (install the launcher by following the [Quickstart](../quickstart.md))

## Tutorials

### [Local Evaluation of Existing Endpoint](tutorials/local-evaluation-of-existing-endpoint.md)

Learn how to evaluate an existing API endpoint using the Local executor. This tutorial covers:

- Choosing model
- Choosing tasks
- Setting up API keys
  - For NVIDIA APIs, see [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)
- Creating configuration file
- Running evaluations

## Quick Start

For detailed step-by-step instructions, see the tutorials above. Here's a quick overview:

### Evaluate existing endpoint

```bash
# Run evaluation
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o target.api_endpoint.api_key=API_KEY
```

## Environment Variables

The Local executor supports passing environment variables from your local machine to evaluation containers:

### How It Works

Environment variables are passed to Docker containers using `docker run -e KEY=VALUE` flags. The executor automatically adds `$` to your variable names from the config `env_vars` (e.g., `OPENAI_API_KEY` becomes `$OPENAI_API_KEY`).

### Configuration

```yaml
evaluation:
  env_vars:
    API_KEY: YOUR_API_KEY_ENV_VAR_NAME
    CUSTOM_VAR: YOUR_CUSTOM_ENV_VAR_NAME
  tasks:
    - name: my_task
      env_vars:
        TASK_SPECIFIC_VAR: TASK_ENV_VAR_NAME
```

## Secrets and API Keys

API keys are handled the same way as environment variables - store them as environment variables on your machine and reference them in the `env_vars` configuration.

## Mounting and Storage

The Local executor uses Docker volume mounts for data persistence:

### Docker Volumes

- **Results Mount**: The `output_dir` is mounted as `/results` in evaluation containers
- **No Custom Mounts**: Local executor doesn't support custom volume mounts (unlike Slurm/Lepton)

## Resuming

The Local executor provides excellent resumption capabilities:

### Script Generation

The Local executor automatically generates scripts for easy resumption:

- **`run_all.sh`**: Master script to run all evaluation tasks (in output directory)
- **`run.sh`**: Individual scripts for each task (in each task subdirectory)
- **Reproducible**: Scripts contain all necessary commands and configurations

### Manual Resumption

```bash
# Resume all tasks
cd /path/to/output_dir/2024-01-15-10-30-45-abc12345/
bash run_all.sh

# Resume specific task
cd /path/to/output_dir/2024-01-15-10-30-45-abc12345/task1/
bash run.sh
```

### Progress Preservation

- **Checkpoint Support**: Evaluations can resume from where they left off
- **No Data Loss**: Already processed samples are preserved

## Key Features

- **Docker-based execution**: Isolated, reproducible runs
- **Existing endpoint evaluation**: Evaluate any OpenAI-compatible endpoint
- **Easy resuming**: Continue interrupted evaluations without losing progress
- **Comprehensive monitoring**: Real-time logs and status tracking

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, see the [Executors Overview](overview.md#job-management) section.
