# Executors Overview

Executors run the evaluation for you by taking the appropriate Docker image (which contains the evaluation harness) and executing the selected benchmark in your environment. They orchestrate containerized runs, manage resources and IO paths, and ensure evaluations are reproducible across machines and clusters. Optionally, an executor can also provision and host the model endpoint as part of the workflow.

- **Core ideas**:
  - Your model is separate from the evaluation container; communication is via an OpenAI‑compatible API
  - Each benchmark runs in an open‑sourced Docker container for reproducibility
  - Execution backends can also optionally manage model deployment

#### Supported execution backends

- `Local`: run on your workstation with Docker
- `Slurm`: submit jobs to an HPC cluster managed by Slurm
- `Lepton`: deploy endpoints and run evaluations on Lepton AI
- `Custom`: build your own executor for any environment (e.g., AWS, Google Cloud, Azure, Kubernetes)

#### Executor Features Comparison

| Feature | Local | Slurm | Lepton |
|---------|-------|-------|--------|
| **Evaluation** | ✅ | ✅ | ✅ |
| **Deployment + Evaluation** | ❌ | ✅ | ✅ |
| **Resuming** | ✅ (Manual) | ✅ (Auto) | ?? |
| **Cloud Native** | ❌ | ❌ | ✅ |
| **Autoscaling** | ❌ | ❌ | ✅ |
| **Best For** | Development, Testing | HPC Clusters | Cloud Scale |

Tip: The simplest way to get started is the Local executor. It pulls the evaluation container to your machine and runs it locally. Your model can live anywhere—as long as it exposes an OpenAI‑compatible endpoint, the local run can call it. See the [Local Evaluation Tutorial](../tutorials/local-evaluation-of-existing-endpoint.md) for a step-by-step guide.

Some executors (e.g., Slurm, Lepton) can optionally host the model on‑the‑fly for the duration of the evaluation. In these cases, two containers are used:

- one for the containerized evaluation (benchmark harness), and
- one for serving the model endpoint.

When on‑the‑fly hosting is enabled, the evaluation configuration also includes a deployment section. See the [deployment configuration documentation](../configuration/deployment/index.md) and examples in the examples/ folder for Slurm and Lepton.

#### Common workflow

1. [Choose an executor and example config](../configuration/index.md)
2. [Point the target to your model endpoint](../configuration/target/index.md)
3. [Run and monitor logs](#job-management)
4. [Optionally export results to dashboards or files](../exporters/overview.md)

#### Key Commands

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

**Job Management:**

```bash
# List runs and monitor
nemo-evaluator-launcher ls runs
nemo-evaluator-launcher status <invocation_id>
nemo-evaluator-launcher kill <invocation_id>
```

**Export Results:**

```bash
# Export to various destinations
nemo-evaluator-launcher export <invocation_id> --dest <local|wandb|mlflow|gsheets>
```

See [Exporters Documentation](../exporters/overview.md) for detailed export options.

#### Test runs

Use a small subset to validate your setup before running full benchmarks:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name <your_config> -o +config.params.limit_samples=10
```

#### Troubleshooting

View the fully resolved configuration without running:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

#### Create your own executor

- Define how to provision resources and launch the evaluation containers
- Implement log and artifact handling
- Expose configuration for resource sizing, networking, and credentials
- Optionally handle model deployment (bring‑up/tear‑down)

See specific guides:

- [Local](local.md)
- [Slurm](slurm.md)
- [Lepton](lepton.md)
- Custom: write your own executor to run evaluations anywhere

## Configuration

For detailed configuration options, see the [Configuration Documentation](../configuration/index.md).

### Output Structure

All executors generate standardized output with timestamped directories. The `output_dir` you specify is mounted as `/results` inside the evaluation container.

**Common across all executors:**

- Configuration saved to `$HOME/.nv-eval/run_configs`
- Timestamped run directories: `output_dir/2024-01-15-10-30-45-abc12345/`
- **Artifacts**: Evaluation results, metrics, predictions, and task-specific outputs
- **Logs**: Execution logs, error messages, and status information

**Example structure:**

```
output_dir/
├── 2024-01-15-10-30-45-abc12345/  # Timestamped run directory
│   ├── ifeval/                     # Task-specific directory
│   │   ├── artifacts/              # Results, metrics, predictions
│   │   └── logs/                   # Execution logs and status
│   └── humaneval_instruct/         # Another task
│       ├── artifacts/
│       └── logs/
```

**Executor-specific scripts:**

- **Local**: `run.sh` (per task) and `run_all.sh` (all tasks) for manual execution
- **Slurm**: `sbatch_*.sh` scripts with HPC job metadata

For detailed output structure and executor-specific details, see the [nemo-evaluator-launcher documentation](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/tree/main/nemo_evaluator_launcher).

## Configuration Files

See all available execution configurations: [Execution Configs](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/tree/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs/execution?ref_type=heads)
