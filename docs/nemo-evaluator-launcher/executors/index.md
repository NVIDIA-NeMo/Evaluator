# Executors Overview

Executors run evaluations by using the appropriate Docker image (which contains the evaluation harness) and executing the selected benchmark in your environment. They orchestrate containerized runs, manage resources and I/O paths, and ensure that evaluations are reproducible across machines and clusters. An executor can also provision and host the model endpoint as part of the workflow.

**Core Ideas**:
  - The model is separate from the evaluation container; communication is by an OpenAI‑compatible API.
  - Each benchmark runs in an open‑source Docker container for reproducibility.
  - Execution back ends can also manage model deployment.

## Supported Execution Back Ends

- `Local`: Run on your workstation with Docker.
- `Slurm`: Submit jobs to an HPC cluster managed by Slurm.
- `Lepton`: Deploy endpoints and run evaluations on Lepton AI.
- `Custom`: Build your own executor for any environment, such as AWS, Google Cloud, Azure, or Kubernetes.

## Executor Feature Comparison

| Feature | Local | Slurm | Lepton |
|---------|-------|-------|--------|
| **Evaluation** | ✅ | ✅ | ✅ |
| **Deployment + Evaluation** | ❌ | ✅ | ✅ |
| **Resuming** | ✅ (Manual) | ✅ (Automatic) | — |
| **Cloud‑native** | ❌ | ❌ | ✅ |
| **Automatic scaling** | ❌ | ❌ | ✅ |
| **Best For** | Development and testing | HPC clusters | Cloud scale |

Tip: The simplest way to get started is the Local executor. It pulls the evaluation container to your machine and runs it locally. The model can be hosted anywhere. As long as it exposes an OpenAI‑compatible endpoint, the local run can call it. Refer to the [Local Evaluation Tutorial](../tutorials/local-evaluation-of-existing-endpoint.md) for a step‑by‑step guide.

Some executors, such as Slurm and Lepton, host the model on the fly for the duration of the evaluation. This workflow uses two containers:

- One for the containerized evaluation (benchmark harness)
- One for serving the model endpoint

When you enable on‑the‑fly hosting, the evaluation configuration also includes a deployment section. Refer to the [Deployment Configuration](../configuration/deployment/index.md) and the examples in the `examples/` folder for Slurm and Lepton.

## Common Workflow

1. [Choose an executor and example configuration](../configuration/index.md)
2. [Point the target to your model endpoint](../configuration/target/index.md)
3. [Run and view logs](#job-management)
4. [Optionally export results to dashboards or files](../exporters/overview.md)

## Key Commands

**Discovery:**

```bash
# List available benchmarks
nemo-evaluator-launcher ls tasks
```

**Execution:**

```bash
# Run evaluations
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct
nemo-evaluator-launcher run --config-dir examples --config-name slurm_llama_3_1_8b_instruct
```

## Job Management

```bash
# List runs and monitor
nemo-evaluator-launcher ls runs
nemo-evaluator-launcher status <invocation_id>
nemo-evaluator-launcher kill <invocation_id>
```

**Export results:**

```bash
# Export to various destinations
nemo-evaluator-launcher export <invocation_id> --dest <local|wandb|mlflow|gsheets>
```
Refer to the [Exporters](../exporters/overview.md) for detailed export options.

## Test Runs

Use a small subset to confirm your setup before running full benchmarks:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name <your_config> -o +config.params.limit_samples=10
```

## Troubleshooting

View the fully resolved configuration without running:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

## Create Your Own Executor

- Define how to provision resources and launch the evaluation containers.
- Handle logs and artifacts.
- Expose configuration for resource sizing, networking, and credentials.
- Optionally handle model deployment (bring‑up and tear‑down).

Refer to the following guides:

- [Local](local.md)
- [Slurm](slurm.md)
- [Lepton](lepton.md)
- Custom: Write your own executor to run evaluations anywhere.

## Configuration

For detailed configuration options, refer to the [Configuration](../configuration/index.md).

### Output Structure

All executors generate standardized output with time‑stamped directories. The evaluation container mounts `output_dir` as `/results`.

**Common across all executors:**

- Configuration saved to `$HOME/.nemo-evaluator-launcher/run_configs`.
- Time‑stamped run directories: `output_dir/2024-01-15-10-30-45-abc12345/`.
- **Artifacts**: Evaluation results, metrics, predictions, and task‑specific outputs.
- **Logs**: Execution logs, error messages, and status information.

**Example structure:**

```text
output_dir/
├── 2024-01-15-10-30-45-abc12345/  # Timestamped run directory
│   ├── ifeval/                     # Task-specific directory
│   │   ├── artifacts/              # Results, metrics, predictions
│   │   └── logs/                   # Execution logs and status
│   └── humaneval_instruct/         # Another task
│       ├── artifacts/
│       └── logs/
```

**Executor‑specific scripts:**

- **Local**: `run.sh` (per task) and `run_all.sh` (all tasks) for manual execution.
- **Slurm**: `sbatch_*.sh` scripts with HPC job metadata.

For detailed output structure and executor‑specific details, refer to the `nemo-evaluator-launcher` package documentation at `packages/nemo-evaluator-launcher`.

## Configuration Files

See all available execution configurations: [Execution Configurations](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs/execution).

