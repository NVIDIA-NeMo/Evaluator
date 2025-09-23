# Local Executor

The Local executor runs evaluations on your machine using Docker. With Docker installed, you can evaluate existing endpoints.

Refer to the [Executors Overview](overview.md) for common concepts and commands.

## Prerequisites

- Docker
- Python environment with the NeMo Evaluator Launcher CLI (refer to the [Tutorial](../tutorial.md) to install)


## Tutorials

### [Local Evaluation of Existing Endpoint](../tutorials/local-evaluation-of-existing-endpoint.md)
Learn how to evaluate an existing API endpoint using the Local executor. This tutorial covers:
- Choose the model
- Choose tasks
- Set up API keys
  - For NVIDIA API keys, refer to [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)
- Create the configuration file
- Run evaluations


## Quick Start

For detailed step-by-step instructions, refer to the tutorials above. Here is a quick overview:

### Run an Existing Endpoint Evaluation

```bash
# Run evaluation
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o target.api_endpoint.api_key=<API_KEY>
```

## Environment Variables

The Local executor supports passing environment variables from your local machine to evaluation containers:

### How It Works

The executor passes environment variables to Docker containers using `docker run -e KEY=VALUE`. It resolves variables from the `env_vars` configuration by prefixing them with `$` when constructing the Docker command (for example, `OPENAI_API_KEY` becomes `$OPENAI_API_KEY`).

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

Handle API keys the same way as environment variables — store them as environment variables on your machine and reference them in the `env_vars` configuration.


## Mounting and Storage

The Local executor uses Docker volume mounts for data persistence:

### Docker Volumes

- **Results Mount**: The executor mounts `output_dir` as `/results` in evaluation containers
- **No Custom Mounts**: The Local executor does not support custom volume mounts (unlike SLURM or Lepton)


## Resuming

The Local executor supports resuming evaluations:

### Script Generation

The Local executor automatically generates scripts to support resumption:
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
- **Existing endpoint support**: Supports any OpenAI-compatible endpoint
- **Easy resuming**: Continue interrupted evaluations without losing progress
- **Comprehensive monitoring**: Real-time logs and status tracking

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, refer to the [Executors Overview](overview.md#job-management) section.