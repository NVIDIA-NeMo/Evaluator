# CLI Reference

The NeMo Evaluator Launcher provides a command-line interface for running evaluations, managing jobs, and exporting results. The CLI is available through two commands:

- `nv-eval` (short alias, recommended)
- `nemo-evaluator-launcher` (full command name)

## Global Options

```bash
nv-eval --help                    # Show help
nv-eval --version                 # Show version information
```

## Commands Overview

| Command | Description |
|---------|-------------|
| `run` | Run evaluations with specified configuration |
| `status` | Check status of jobs or invocations |
| `kill` | Kill a job or invocation |
| `ls` | List tasks or runs |
| `export` | Export evaluation results to various destinations |
| `version` | Show version information |

## run - Run Evaluations

Execute evaluations using Hydra configuration management.

### Basic Usage

```bash
# Using example configurations
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct

# With output directory override
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=/path/to/results
```

### Configuration Options

```bash
# Using custom config directory
nv-eval run --config-dir my_configs --config-name my_evaluation

# Multiple overrides (Hydra syntax)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=results \
  -o target.api_endpoint.model_id=my-model \
  -o +config.params.limit_samples=10
```

### Dry Run

Preview the full resolved configuration without executing:

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

### Test Runs

Run with limited samples for testing:

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```

### Examples by Executor

**Local Execution:**

```bash
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=./local_results
```

**Slurm Execution:**

```bash
nv-eval run --config-dir examples --config-name slurm_llama_3_1_8b_instruct \
  -o execution.output_dir=/shared/results
```

**Lepton AI Execution:**

```bash
# With model deployment
nv-eval run --config-dir examples --config-name lepton_nim_llama_3_1_8b_instruct

# Using existing endpoint
nv-eval run --config-dir examples --config-name lepton_none_llama_3_1_8b_instruct
```

## status - Check Job Status

Check the status of running or completed evaluations.

### Status Basic Usage

```bash
# Check status of specific invocation (returns all jobs in that invocation)
nv-eval status abc12345

# Check status of specific job
nv-eval status abc12345.0

# Output as JSON
nv-eval status abc12345 --json
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

## kill - Kill Jobs

Stop running evaluations.

### Kill Basic Usage

```bash
# Kill entire invocation
nv-eval kill abc12345

# Kill specific job
nv-eval kill abc12345.0
```

The command outputs JSON with the results of the kill operation.

## ls - List Resources

List available tasks or runs.

### List Tasks

```bash
# List all available evaluation tasks
nv-eval ls tasks

# List tasks with JSON output
nv-eval ls tasks --json
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
nv-eval ls runs

# Limit number of results
nv-eval ls runs --limit 10

# Filter by executor
nv-eval ls runs --executor local

# Filter by date
nv-eval ls runs --since "2024-01-01"
nv-eval ls runs --since "2024-01-01T12:00:00"
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
nv-eval export abc12345 --dest local --format json

# Export to specific directory
nv-eval export abc12345 --dest local --format json --output-dir ./results

# Specify custom filename
nv-eval export abc12345 --dest local --format json --output-filename results.json
```

### Export Options

```bash
# Available destinations
nv-eval export abc12345 --dest local      # Local file system
nv-eval export abc12345 --dest mlflow     # MLflow tracking
nv-eval export abc12345 --dest wandb      # Weights & Biases
nv-eval export abc12345 --dest gsheets    # Google Sheets
nv-eval export abc12345 --dest jet        # JET (internal)

# Format options (for local destination only)
nv-eval export abc12345 --dest local --format json
nv-eval export abc12345 --dest local --format csv

# Include logs when exporting
nv-eval export abc12345 --dest local --format json --copy-logs

# Filter metrics by name
nv-eval export abc12345 --dest local --format json --log-metrics score --log-metrics accuracy

# Copy all artifacts (not just required ones)
nv-eval export abc12345 --dest local --only-required False
```

### Exporting Multiple Invocations

```bash
# Export several runs together
nv-eval export abc12345 def67890 ghi11111 --dest local --format json

# Export several runs with custom output
nv-eval export abc12345 def67890 --dest local --format csv \
  --output-dir ./all-results --output-filename combined.csv
```

### Cloud Exporters

For cloud destinations like MLflow, W&B, and Google Sheets, configure credentials through environment variables or their respective configuration files before using the export command. Refer to each exporter's documentation for setup instructions.

## version - Version Information

Display version and build information.

```bash
# Show version
nv-eval version

# Alternative
nv-eval --version
```

## Environment Variables

The CLI respects environment variables for logging and task-specific authentication:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEMO_EVALUATOR_LOG_LEVEL` | Logging level for the launcher (DEBUG, INFO, WARNING, ERROR, CRITICAL) | `WARNING` |
| `LOG_LEVEL` | Alternative log level variable | Uses `NEMO_EVALUATOR_LOG_LEVEL` if set |
| `LOG_DISABLE_REDACTION` | Disable credential redaction in logs (set to 1, true, or yes) | Not set |

### Task-Specific Environment Variables

Some evaluation tasks require API keys or tokens. These are configured in your evaluation YAML file under `env_vars` and must be set before running:

```bash
# Set task-specific environment variables
export HF_TOKEN="hf_..."              # For Hugging Face datasets
export API_KEY="nvapi-..."            # For NVIDIA API endpoints

# Run evaluation
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

The specific environment variables required depend on the tasks and endpoints you're using. Refer to the example configuration files for details on which variables are needed.

## Configuration File Examples

The NeMo Evaluator Launcher includes several example configuration files that demonstrate different use cases. These files are located in the `examples/` directory of the package:

- `local_llama_3_1_8b_instruct.yaml` - Local execution with an existing endpoint
- `local_limit_samples.yaml` - Local execution with limited samples for testing
- `slurm_llama_3_1_8b_instruct.yaml` - Slurm execution with model deployment
- `slurm_no_deployment_llama_3_1_8b_instruct.yaml` - Slurm execution with existing endpoint
- `lepton_nim_llama_3_1_8b_instruct.yaml` - Lepton AI execution with NIM deployment
- `lepton_vllm_llama_3_1_8b_instruct.yaml` - Lepton AI execution with vLLM deployment
- `lepton_none_llama_3_1_8b_instruct.yaml` - Lepton AI execution with existing endpoint

To use these examples:

```bash
# Copy an example to your local directory
cp examples/local_llama_3_1_8b_instruct.yaml my_config.yaml

# Edit the configuration as needed
# Then run with your config
nv-eval run --config-dir . --config-name my_config
```

Refer to the {ref}`configuration documentation <configuration-overview>` for detailed information on all available configuration options.

## Troubleshooting

### Configuration Issues

**Configuration Errors:**

```bash
# Validate configuration without running
nv-eval run --config-dir examples --config-name my_config --dry-run
```

**Permission Errors:**

```bash
# Check file permissions
ls -la examples/my_config.yaml

# Use absolute paths
nv-eval run --config-dir /absolute/path/to/configs --config-name my_config
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
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct

# Or use single-letter shorthand
export NEMO_EVALUATOR_LOG_LEVEL=D
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct

# Logs are written to ~/.nemo-evaluator/logs/
```

### Getting Help

```bash
# Command-specific help
nv-eval run --help
nv-eval export --help
nv-eval ls --help

# General help
nv-eval --help
```

## See Also

- [Python API](api.md) - Programmatic interface
- {ref}`launcher-quickstart` - Getting started guide
- {ref}`executors-overview` - Execution backends
- {ref}`exporters-overview` - Export destinations
