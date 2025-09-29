# Slurm Executor

The Slurm executor runs evaluations on high‑performance computing (HPC) clusters managed by Slurm, an open‑source workload manager widely used in research and enterprise environments. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility by using containerized benchmarks.

Refer to common concepts and commands in the [Executors Overview](overview.md).

Slurm can optionally host your model for the scope of an evaluation by deploying a serving container on the cluster and pointing the benchmark to that temporary endpoint. In this mode, the system uses two containers: one for the evaluation harness and one for the model server. The evaluation configuration includes a `deployment` section when enabled. Refer to the examples in the `examples/` directory for ready‑to‑use configurations.

If you do not require deployment on Slurm, omit the `deployment` section from your configuration and set the model endpoint URL directly (any OpenAI‑compatible endpoint that you host elsewhere).

## Prerequisites

Ensure the following:

- Access to a Slurm cluster (with appropriate partitions or queues)
- Docker or other container runtime available on worker nodes
- Define the Slurm executor configuration before running evaluations

## Environment Variables

The Slurm executor supports environment variables through `execution.env_vars`:

### Configuration

```yaml
execution:
  env_vars:
    deployment:
      CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
      USER: ${oc.env:USER}  # References host environment variable
    evaluation:
      CUSTOM_VAR: YOUR_CUSTOM_ENV_VAR_NAME
evaluation:
  env_vars:
    CUSTOM_VAR: YOUR_CUSTOM_ENV_VAR_NAME
  tasks:
    - name: my_task
      env_vars:
        TASK_SPECIFIC_VAR: TASK_ENV_VAR_NAME
```

### Key Points

- **Deployment Variables**: Use `execution.env_vars.deployment` for model serving containers.
- **Evaluation Variables**: Use `execution.env_vars.evaluation` for evaluation containers.
- **Direct Values**: Use quoted strings for direct values.
- **Hydra Environment Variables**: Use `${oc.env:VARIABLE_NAME}` to reference host environment variables.
- **Environment Variable Names**: Use the names of environment variables on your local machine.

## Secrets and API Keys

Handle API keys the same way as environment variables. Store them as environment variables on your machine and reference them in the `execution.env_vars` configuration.

### Security Considerations

- **Do Not Hard‑Code**: Do not put API keys in configuration files.
- **SSH Security**: Configure Secure Shell (SSH) securely for key transmission to the cluster.
- **File Permissions**: Set appropriate file permissions (not world‑readable).
- **Public Clusters**: The batch script stores secrets from `execution.env_vars` in plain text and saves them under `output_dir` on the login node. Use caution when handling sensitive data on public clusters.

## Mounting and Storage

The Slurm executor provides mounting capabilities:

### Mount Configuration

```yaml
execution:
  mounts:
    deployment:
      /path/to/checkpoints: /checkpoint
      /path/to/cache: /cache
    evaluation:
      /path/to/data: /data
      /path/to/results: /results
    mount_home: true  # Mount user home directory
```

### Mount Types

- **Deployment Mounts**: For model checkpoints, cache directories, and model data.
- **Evaluation Mounts**: For input data, results, and evaluation-specific files.
- **Home Mount**: Optional mounting of user home directory (enabled by default).

## Resuming

The Slurm executor includes advanced auto-resume capabilities:

### Automatic Resumption

- **Timeout Handling**: Jobs automatically resume after a timeout.
- **Preemption Recovery**: Jobs automatically resume after preemption.
- **Node Failure Recovery**: Jobs automatically resume after node failures.
- **Dependency Management**: Uses Slurm job dependencies for resumption.

### How It Works

1. **Initial Submission**: Submit the job with the auto‑resume handler.
2. **Failure Detection**: The script detects a timeout, preemption, or failure.
3. **Automatic Resubmission**: Submit a new job with a dependency on the previous job.
4. **Progress Preservation**: Continue the evaluation from where it left off.

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, refer to the [Executors Overview](index.md#job-management) section.
