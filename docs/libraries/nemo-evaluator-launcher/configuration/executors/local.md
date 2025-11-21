(executor-local)=

# Local Executor

The Local executor runs evaluations on your machine using Docker. It provides a fast way to iterate if you have Docker installed, evaluating existing endpoints.

See common concepts and commands in {ref}`executors-overview`.

## Prerequisites

- Docker
- Python environment with the NeMo Evaluator Launcher CLI available (install the launcher by following {ref}`gs-install`)

## Quick Start

For detailed step-by-step instructions on evaluating existing endpoints, refer to the {ref}`gs-quickstart-launcher` guide, which covers:

- Choosing models and tasks
- Setting up API keys (for NVIDIA APIs, see [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key))
- Creating configuration files
- Running evaluations

Here's a quick overview for the Local executor:

### Run evaluation for existing endpoint

```bash
# Run evaluation
nemo-evaluator-launcher run --config packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml \
  -o target.api_endpoint.api_key_name=NGC_API_KEY
```

## Environment Variables

The Local executor supports passing environment variables from your local machine to evaluation containers:

### How It Works

The executor passes environment variables to Docker containers using `docker run -e KEY=VALUE` flags. The executor automatically adds `$` to your variable names from the configuration `env_vars` (for example, `OPENAI_API_KEY` becomes `$OPENAI_API_KEY`).

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

The executor handles API keys the same way as environment variables - store them as environment variables on your machine and reference them in the `env_vars` configuration.

## Mounting and Storage

The Local executor uses Docker volume mounts for data persistence:

### Docker Volumes

- **Results Mount**: Each task's artifacts directory mounts as `/results` in evaluation containers
- **Custom Mounts**: Use to `extra_docker_args` field to define custom volume mounts (see [Advanced configuration](#advanced-configuration) )

## Advanced configuration

You can customize your local executor by specifying `extra_docker_args`.
This parameter allows you to pass any flag to the `docker run` command that is executed by the NeMo Evaluator Launcher.
You can use it to mount additional volumes, set environment variables or customize your network settings.

For example, if you would like your job to use a specific docker network, you can specify:

```yaml
execution:
  extra_docker_args: "--network my-custom-network"
```

Replace `my-custom-network` with `host` to access the host network.

To mount additional custom volumes, do:

```yaml
execution:
  extra_docker_args: "--volume /my/local/path:/my/container/path"
```


## Rerunning Evaluations

The Local executor generates reusable scripts for rerunning evaluations:

### Script Generation

The Local executor automatically generates scripts:

- **`run_all.sequential.sh`**: Script to run all evaluation tasks sequentially (in output directory)
- **`run.sh`**: Individual scripts for each task (in each task subdirectory)
- **Reproducible**: Scripts contain all necessary commands and configurations

### Manual Rerun

```bash
# Rerun all tasks
cd /path/to/output_dir/2024-01-15-10-30-45-abc12345/
bash run_all.sequential.sh

# Rerun specific task
cd /path/to/output_dir/2024-01-15-10-30-45-abc12345/task1/
bash run.sh
```

## Key Features

- **Docker-based execution**: Isolated, reproducible runs
- **Script generation**: Reusable scripts for rerunning evaluations
- **Real-time logs**: Status tracking via log files

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, see {ref}`executors-overview`.
