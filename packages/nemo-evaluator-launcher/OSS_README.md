# NV-Eval

> **Submit bugs**: please help us improve by submitting bugs and improvements http://nv/eval.issue!

A comprehensive evaluation platform for large language models (LLMs) that supports multiple benchmarks and execution environments. NV-Eval provides both a command-line interface for rapid experimentation and a Python library interface (`nemo_evaluator_launcher.api`) for programmatic integration into research workflows.

## Purpose

NV-Eval is designed to enable fully transparent and reproducible experiment setups for LLM evaluation. The platform achieves this by:

- **Open Source Docker Containers**: All benchmarks run in open-sourced Docker containers, ensuring the exact same system setup regardless of who runs the experiments
- **Reproducible Configurations**: Complete experiment configurations are saved and can be rerun with perfect reproducibility
- **Transparent Evaluation**: Every aspect of the evaluation process is documented and accessible
- **Consistent Environments**: Eliminates "works on my machine" issues by using standardized containerized environments

This approach ensures that LLM evaluation results are comparable across different research groups, institutions, and time periods, making the field more reliable and trustworthy.

## Architecture

NV-Eval manages the containers that run evaluations and supports 38+ benchmarks 14+ harnesses. The platform follows a clean separation of concerns:

### Model Evaluation Setup

The model you are evaluating is completely separate from the container running the evaluation. The evaluation container communicates with your model through an OpenAI-compatible API and has zero knowledge of the model's deployment details. This design allows you to evaluate any model that exposes a compatible API endpoint, regardless of how it's deployed or hosted.

### Execution Environments

NV-Eval supports multiple execution environments to fit your infrastructure:

- **Local Execution**: Run evaluations locally if you have Docker installed
- **Slurm Clusters**: Execute experiments on HPC clusters using Slurm
- **Custom Executors**: Extend the platform with your own executors for Kubernetes, AWS, Google Cloud, or other cloud platforms

### Deployment Integration

Executors can optionally support model deployment, allowing you to both deploy your model and run evaluations with a single command. This is useful when the model you want to evaluate isn't already served and available under an API endpoint.

## Installation

Install the package using pip:

**TODO(public release)**: replace the install script

```bash
pip install nemo-evaluator-launcher-internal --index-url https://gitlab-master.nvidia.com/api/v4/projects/155749/packages/pypi/simple
```

## Quick Start

### 1. List Available Benchmarks

View all available evaluation benchmarks:

```bash
nv-eval ls
```

**TODO(public release): replace this with the README of the `nemo-evaluator`.

Containers and exact run commands can be found [here](https://nv-eval-platform-dl-joc-competitive-evaluation-c0c268c1aead4fd0.gitlab-master-pages.nvidia.com/benchmarks_doc/).

### 2. Run Evaluations

NV-Eval uses a configuration-based experiment setup powered by Hydra. Each configuration specifies exactly what container to use and fixes all parameters of the experiment, ensuring perfect reproducibility. When running experiments with NV-Eval, each past run's complete configuration is automatically saved under `~/.nv-eval/run_configs`, allowing you to rerun experiments with the exact same settings or share them with others for collaboration.

#### Invocation Output and Monitoring

When you start an evaluation, NV-Eval provides you with an invocation ID and helpful commands for monitoring:

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

**Example output:**
```
Commands for real-time monitoring:
  tail -f /path/to/output/005b17d3/gpqa_diamond/logs/stdout.log

Follow all logs for this invocation:
  tail -f /path/to/output/005b17d3/*/logs/stdout.log

Complete run config saved to: /Users/username/.nv-eval/run_configs/005b17d3_config.yml
to check status: nv-eval status 005b17d3
to kill all jobs: nv-eval kill 005b17d3
to kill individual jobs: nv-eval kill <job_id> (e.g., 005b17d3.0)
```

**Live Log Monitoring:**

The recommended way to follow evaluation progress is using the `tail -f` command. Since NV-Eval runs multiple benchmark evaluations in parallel, there's no single console output that can effectively display all logs simultaneously without becoming overwhelming.

**Recommended approaches:**

1. **Follow a specific task**: Use the individual task log path to focus on one benchmark at a time
   ```bash
   tail -f /path/to/output/005b17d3/gpqa_diamond/logs/stdout.log
   ```

2. **Follow all tasks in parallel**: Use the wildcard path to see all parallel evaluations mixed together. This can be useful when you want to monitor overall progress across all benchmarks:
   ```bash
   tail -f /path/to/output/005b17d3/*/logs/stdout.log
   ```

3. **Open multiple terminals**: For detailed monitoring, open separate terminal windows for each task you want to track closely.

#### Using Example Configurations

The [examples/](examples/) directory contains ready-to-use configurations:

- **Local execution**: [local_llama_3_1_8b_instruct.yaml](examples/local_llama_3_1_8b_instruct.yaml)
- **Slurm cluster execution**: [slurm_llama_3_1_8b_instruct.yaml](examples/slurm_llama_3_1_8b_instruct.yaml)

Run a local evaluation:
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

Run a Slurm cluster evaluation:
```bash
nv-eval run --config-dir examples --config-name slurm_llama_3_1_8b_instruct execution.output_dir=<YOUR_OUTPUT_DIR_ON_CLUSTER>
```

Generate all the configs. Useful for: seeing full examples of merged configurations, avoiding Hydra indirections.
```bash
python scripts/generate_configs.py
```

#### Creating Custom Configurations

1. Create your own configuration directory:
```bash
mkdir my_configs
```

2. Copy an example configuration as a starting point:
```bash
cp examples/local_llama_3_1_8b_instruct.yaml my_configs/my_evaluation.yaml
```

3. Modify the configuration to suit your needs:
   - Change the model endpoint
   - Adjust evaluation parameters
   - Select different benchmarks
   - Configure execution settings

4. Run your custom configuration:
```bash
nv-eval run --config-dir my_configs --config-name my_evaluation
```

#### Configuration Overrides

You can override configuration values from the command line (`-o` can be used multiple times, the notation is following hydra)

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

#### Test Runs

All benchmarks support test runs, which are useful for validating your configuration and ensuring everything works end-to-end before running full evaluations. Test runs use only a subset of the benchmark data, making them much faster to complete.

**To run a test evaluation:**

1. **Using configuration overrides:**
   ```bash
   nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
     -o +config.params.limit_samples=10
   ```

2. **Using a test configuration file:**
   ```bash
   nv-eval run --config-dir examples --config-name local_limit_samples
   ```

The `limit_samples` parameter controls how many samples to evaluate. For example:
- `limit_samples: 10` - evaluates only 10 samples in total from the benchmark

**Important:** Results from test runs should **never be used** for benchmarking or comparison purposes. Test runs are solely for:
- Validating your configuration works correctly
- Testing end-to-end functionality
- Debugging setup issues

Always run full evaluations (without `limit_samples`) for actual benchmark results.

### Key Commands

NV-Eval provides various commands for managing evaluations. Use the `-h` flag to list all available commands:

```bash
nv-eval -h
```

**Check job status:**
```bash
nv-eval status <invocation_id>     # Check all jobs in an invocation
nv-eval status <job_id>            # Check specific job status
```

**List available benchmarks:**
```bash
nv-eval ls                         # List all available benchmarks
```

**Run evaluations:**
```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

**Manage running jobs:**
```bash
nv-eval kill <invocation_id>       # Kill all jobs in an invocation
nv-eval kill <job_id>              # Kill a specific job
```

## Python API

Consider checking out [Python notebooks](./examples/notebooks)

## Troubleshooting

### View Full Configuration

To see the complete resolved configuration:

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

## Contributing

**TODO(public release)**: Add contributing part
