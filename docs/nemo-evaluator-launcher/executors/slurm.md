# Slurm Executor

The Slurm executor runs evaluations on high‑performance computing (HPC) clusters managed by Slurm, an open‑source workload manager widely used in research and enterprise environments. It schedules and executes jobs across cluster nodes, enabling parallel, large‑scale evaluation runs while preserving reproducibility via containerized benchmarks.

See common concepts and commands in the [executors overview](overview.md).

Slurm can optionally host your model for the scope of an evaluation by deploying a serving container on the cluster and pointing the benchmark to that temporary endpoint. In this mode, two containers are used: one for the evaluation harness and one for the model server. The evaluation configuration includes a deployment section when this is enabled. See the examples in the examples/ directory for ready‑to‑use configurations.

If you do not require deployment on Slurm, simply omit the deployment section from your configuration and set the model’s endpoint URL directly (any OpenAI‑compatible endpoint that you host elsewhere).

## Prerequisites

- Access to a Slurm cluster (with appropriate partitions/queues)
- Docker or container runtime available on worker nodes (per your environment)
- Slurm cluster configuration must be defined in the executors configuration before running evaluations

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

- **Deployment Variables**: Use `execution.env_vars.deployment` for model serving containers
- **Evaluation Variables**: Use `execution.env_vars.evaluation` for evaluation containers
- **Direct Values**: Use quoted strings for direct values
- **Hydra Environment Variables**: Use `${oc.env:VARIABLE_NAME}` to reference host environment variables
- **Environment Variable Names**: Use the names of environment variables on your local machine

## Secrets and API Keys

API keys are handled the same way as environment variables - store them as environment variables on your machine and reference them in the `execution.env_vars` configuration.

### Security Considerations

- **No Hardcoding**: Never put API keys directly in configuration files
- **SSH Security**: Ensure secure SSH configuration for key transmission to the cluster
- **File Permissions**: Ensure configuration files have appropriate permissions (not world-readable)
- **Public Clusters**: Secrets in `execution.env_vars` are stored in plain text in the batch script and saved under `output_dir` on the login node. Use caution when handling sensitive data on public clusters.

## Mounting and Storage

The Slurm executor provides sophisticated mounting capabilities:

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

- **Deployment Mounts**: For model checkpoints, cache directories, and model data
- **Evaluation Mounts**: For input data, results, and evaluation-specific files
- **Home Mount**: Optional mounting of user home directory (enabled by default)

## Resuming

The Slurm executor includes advanced auto-resume capabilities:

### Automatic Resumption

- **Timeout Handling**: Jobs automatically resume after timeout
- **Preemption Recovery**: Automatic resumption after job preemption
- **Node Failure Recovery**: Jobs resume after node failures
- **Dependency Management**: Uses Slurm job dependencies for resumption

### How It Works

1. **Initial Submission**: Job is submitted with auto-resume handler
2. **Failure Detection**: Script detects timeout/preemption/failure
3. **Automatic Resubmission**: New job is submitted with dependency on previous job
4. **Progress Preservation**: Evaluation continues from where it left off

## Monitoring and Job Management

For monitoring jobs, checking status, and managing evaluations, see the [Executors Overview](overview.md#job-management) section.
