# NeMo Evaluator Launcher CLI Reference (nemo-evaluator-launcher)

The NeMo Evaluator Launcher provides a command-line interface for running evaluations, managing jobs, and exporting results. The CLI is available through `nemo-evaluator-launcher` command.

## Global Options

```bash
nemo-evaluator-launcher --help                    # Show help
nemo-evaluator-launcher --version                 # Show version information
```

## Commands Overview

```{list-table}
:header-rows: 1
:widths: 20 80

* - Command
  - Description
* - `run`
  - Run evaluations with specified configuration
* - `status`
  - Check status of jobs or invocations
* - `info`
  - Show detailed job(s) information
* - `kill`
  - Kill a job or invocation
* - `ls`
  - List tasks or runs
* - `export`
  - Export evaluation results to various destinations
* - `version`
  - Show version information
```

## run - Run Evaluations

Execute evaluations using Hydra configuration management.

### Basic Usage

```bash
# Using example configurations
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct

# With output directory override
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=/path/to/results
```

### Configuration Options

```bash
# Using custom config directory
nemo-evaluator-launcher run --config-dir my_configs --config-name my_evaluation

# Multiple overrides (Hydra syntax)
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=results \
  -o target.api_endpoint.model_id=my-model \
  -o +config.params.limit_samples=10
```

(launcher-cli-dry-run)=
### Dry Run

Preview the full resolved configuration without executing:

```bash
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct --dry-run
```

### Test Runs

Run with limited samples for testing:

```bash
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```

### Examples by Executor

**Local Execution:**

```bash
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=./local_results
```

**Slurm Execution:**

```bash
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name slurm_llama_3_1_8b_instruct \
  -o execution.output_dir=/shared/results
```

**Lepton AI Execution:**

```bash
# With model deployment
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name lepton_nim_llama_3_1_8b_instruct

# Using existing endpoint
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name lepton_none_llama_3_1_8b_instruct
```

## status - Check Job Status

Check the status of running or completed evaluations.

### Status Basic Usage

```bash
# Check status of specific invocation (returns all jobs in that invocation)
nemo-evaluator-launcher status abc12345

# Check status of specific job
nemo-evaluator-launcher status abc12345.0

# Output as JSON
nemo-evaluator-launcher status abc12345 --json
```

### Output Formats

**Table Format (default):**

```text
Job ID       | Status   | Executor Info | Location
abc12345.0   | running  | container123  | <output_dir>/task1/...
abc12345.1   | success  | container124  | <output_dir>/task2/...
```

**JSON Format (with --json flag):**

```json
[
  {
    "invocation": "abc12345",
    "job_id": "abc12345.0",
    "status": "running",
    "data": {
      "container": "eval-container",
      "output_dir": "/path/to/results"
    }
  },
  {
    "invocation": "abc12345",
    "job_id": "abc12345.1",
    "status": "success",
    "data": {
      "container": "eval-container",
      "output_dir": "/path/to/results"
    }
  }
]
```

## info - Job information and navigation

Display detailed job information, including metadata, configuration, and paths to logs/artifacts with descriptions of key result files. Supports copying results locally from both local and remote jobs.

### Basic usage
```bash
# Show job info for one or more IDs (job or invocation)
nemo-evaluator-launcher info <job_or_invocation_id>
nemo-evaluator-launcher info <inv1> <inv2>
```

### Show configuration
```bash
nemo-evaluator-launcher info <id> --config
```

### Show paths
```bash
# Show artifact locations
nemo-evaluator-launcher info <id> --artifacts
# Show log locations
nemo-evaluator-launcher info <id> --logs
```

### Copy files locally
```bash
# Copy logs
nemo-evaluator-launcher info <id> --copy-logs [DIR]

# Copy artifacts
nemo-evaluator-launcher info <id> --copy-artifacts [DIR]
```

### Example (Slurm)
```text
nemo-evaluator-launcher info <inv_id>

Job <inv_id>.0
├── Executor: slurm
├── Created: <timestamp>
├── Task: <task_name>
├── Artifacts: user@host:/shared/.../<job_id>/task_name/artifacts (remote)
│   └── Key files:
│       ├── results.yml - Benchmark scores, task results and resolved run configuration.
│       ├── eval_factory_metrics.json - Response + runtime stats (latency, tokens count, memory)
│       ├── metrics.json - Harness/benchmark metric and configuration
│       ├── report.html - Request-Response Pairs samples in HTML format (if enabled)
│       ├── report.json - Report data in json format, if enabled
├── Logs: user@host:/shared/.../<job_id>/task_name/logs (remote)
│   └── Key files:
│       ├── client-{SLURM_JOB_ID}.out - Evaluation container/process output
│       ├── slurm-{SLURM_JOB_ID}.out - SLURM scheduler stdout/stderr (batch submission, export steps).
│       ├── server-{SLURM_JOB_ID}.out - Model server logs when a deployment is used.
├── Slurm Job ID: <SLURM_JOB_ID>
```

## kill - Kill Jobs

Stop running evaluations.

### Kill Basic Usage

```bash
# Kill entire invocation
nemo-evaluator-launcher kill abc12345

# Kill specific job
nemo-evaluator-launcher kill abc12345.0
```

The command outputs JSON with the results of the kill operation.

## ls - List Resources

List available tasks or runs.

### List Tasks

```bash
# List all available evaluation tasks
nemo-evaluator-launcher ls tasks

# List tasks with JSON output
nemo-evaluator-launcher ls tasks --json
```

**Output Format:**

Tasks display grouped by harness and container, showing the task name and required endpoint type:

```text
===================================================
harness: lm_eval
container: nvcr.io/nvidia/nemo:24.01

task                    endpoint_type
---------------------------------------------------
arc_challenge           chat
hellaswag              completions
winogrande             completions
---------------------------------------------------
  3 tasks available
===================================================
```

### List Runs

```bash
# List recent evaluation runs
nemo-evaluator-launcher ls runs

# Limit number of results
nemo-evaluator-launcher ls runs --limit 10

# Filter by executor
nemo-evaluator-launcher ls runs --executor local

# Filter by date
nemo-evaluator-launcher ls runs --since "2024-01-01"
nemo-evaluator-launcher ls runs --since "2024-01-01T12:00:00"

# Filter by retrospecitve period
# - days
nemo-evaluator-launcher ls runs --since 2d
# - hours
nemo-evaluator-launcher ls runs --since 6h
```

**Output Format:**

```text
invocation_id  earliest_job_ts       num_jobs  executor  benchmarks
abc12345       2024-01-01T10:00:00   3         local     ifeval,gpqa_diamond,mbpp
def67890       2024-01-02T14:30:00   2         slurm     hellaswag,winogrande
```

## export - Export Results

Export evaluation results to various destinations.

### Export Basic Usage

```bash
# Export to local files (JSON format)
nemo-evaluator-launcher export abc12345 --dest local --format json

# Export to specific directory
nemo-evaluator-launcher export abc12345 --dest local --format json --output-dir ./results

# Specify custom filename
nemo-evaluator-launcher export abc12345 --dest local --format json --output-filename results.json
```

### Export Options

```bash
# Available destinations
nemo-evaluator-launcher export abc12345 --dest local      # Local file system
nemo-evaluator-launcher export abc12345 --dest mlflow     # MLflow tracking
nemo-evaluator-launcher export abc12345 --dest wandb      # Weights & Biases
nemo-evaluator-launcher export abc12345 --dest gsheets    # Google Sheets

# Format options (for local destination only)
nemo-evaluator-launcher export abc12345 --dest local --format json
nemo-evaluator-launcher export abc12345 --dest local --format csv

# Include logs when exporting
nemo-evaluator-launcher export abc12345 --dest local --format json --copy-logs

# Filter metrics by name
nemo-evaluator-launcher export abc12345 --dest local --format json --log-metrics score --log-metrics accuracy

# Copy all artifacts (not just required ones)
nemo-evaluator-launcher export abc12345 --dest local --only-required False
```

### Exporting Multiple Invocations

```bash
# Export several runs together
nemo-evaluator-launcher export abc12345 def67890 ghi11111 --dest local --format json

# Export several runs with custom output
nemo-evaluator-launcher export abc12345 def67890 --dest local --format csv \
  --output-dir ./all-results --output-filename combined.csv
```

### Cloud Exporters

For cloud destinations like MLflow, W&B, and Google Sheets, configure credentials through environment variables or their respective configuration files before using the export command. Refer to each exporter's documentation for setup instructions.

## version - Version Information

Display version and build information.

```bash
# Show version
nemo-evaluator-launcher version

# Alternative
nemo-evaluator-launcher --version
```

## Environment Variables

The CLI respects environment variables for logging and task-specific authentication:

```{list-table}
:header-rows: 1
:widths: 30 50 20

* - Variable
  - Description
  - Default
* - `LOG_LEVEL`
  - Logging level for the launcher (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  - `WARNING`
* - `LOG_DISABLE_REDACTION`
  - Disable credential redaction in logs (set to 1, true, or yes)
  - Not set
```

### Task-Specific Environment Variables

Some evaluation tasks require API keys or tokens. These are configured in your evaluation YAML file under `env_vars` and must be set before running:

```bash
# Set task-specific environment variables
export HF_TOKEN="hf_..."              # For Hugging Face datasets
export NGC_API_KEY="nvapi-..."            # For NVIDIA API endpoints

# Run evaluation
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct
```

The specific environment variables required depend on the tasks and endpoints you're using. Refer to the example configuration files for details on which variables are needed.

## Configuration File Examples

The NeMo Evaluator Launcher includes several example configuration files that demonstrate different use cases. These files are located in the `examples/` directory of the package:

- `local_llama_3_1_8b_instruct.yaml` - Local execution with an existing endpoint
- `local_limit_samples.yaml` - Local execution with limited samples for testing
- `local_nvidia_nemotron_nano_9b_v2.yaml` - Local execution with Nvidia Nemotron Nano 9B v2
- `local_auto_export_llama_3_1_8b_instruct.yaml` - Local execution with auto-export for Llama 3.1 8B
- `local_custom_config_seed_oss_36b_instruct.yaml` - Local execution with advanced interceptors
- `slurm_llama_3_1_8b_instruct.yaml` - Slurm execution with model deployment
- `slurm_llama_3_1_8b_instruct_hf.yaml` - Slurm execution with deployment using Hugging Face model handle
- `slurm_no_deployment_llama_3_1_8b_instruct.yaml` - Slurm execution with existing endpoint
- `slurm_no_deployment_llama_nemotron_super_v1_nemotron_benchmarks.yaml` - Slurm execution with Llama-3.3-Nemotron-Super
- `lepton_nim_llama_3_1_8b_instruct.yaml` - Lepton AI execution with NIM deployment
- `lepton_vllm_llama_3_1_8b_instruct.yaml` - Lepton AI execution with vLLM deployment
- `lepton_none_llama_3_1_8b_instruct.yaml` - Lepton AI execution with existing endpoint

To use these examples:

```bash
# Copy an example to your local directory
cp examples/local_llama_3_1_8b_instruct.yaml my_config.yaml

# Edit the configuration as needed
# Then run with your config
nemo-evaluator-launcher run --config-dir . --config-name my_config
```

Refer to the {ref}`configuration documentation <configuration-overview>` for detailed information on all available configuration options.

## Troubleshooting

### Configuration Issues

**Configuration Errors:**

```bash
# Validate configuration without running
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name my_config --dry-run
```

**Permission Errors:**

```bash
# Check file permissions
ls -la examples/my_config.yaml

# Use absolute paths
nemo-evaluator-launcher run --config-dir /absolute/path/to/configs --config-name my_config
```

**Network Issues:**

```bash
# Test endpoint connectivity
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "test", "messages": [{"role": "user", "content": "Hello"}]}'
```

### Debug Mode

```bash
# Set log level to DEBUG for detailed output
export LOG_LEVEL=DEBUG
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct

# Or use single-letter shorthand
export LOG_LEVEL=D
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_llama_3_1_8b_instruct

# Logs are written to ~/.nemo-evaluator/logs/
```

### Getting Help

```bash
# Command-specific help
nemo-evaluator-launcher run --help
nemo-evaluator-launcher info --help
nemo-evaluator-launcher ls --help
nemo-evaluator-launcher export --help

# General help
nemo-evaluator-launcher --help
```

## See Also

- [Python API](api.md) - Programmatic interface
- {ref}`gs-quickstart-launcher` - Getting started guide
- {ref}`executors-overview` - Execution backends
- {ref}`exporters-overview` - Export destinations
