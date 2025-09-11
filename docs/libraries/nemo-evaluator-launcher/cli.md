# CLI Reference

The NeMo Evaluator Launcher provides a comprehensive command-line interface for running evaluations, managing jobs, and exporting results. The CLI is available through two commands:

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
| `kill` | Kill running jobs or invocations |
| `ls` | List tasks, runs, or other resources |
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
  --override execution.output_dir=/path/to/results
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
  --override execution.output_dir=./local_results
```

**Slurm Execution:**
```bash
nv-eval run --config-dir examples --config-name slurm_llama_3_1_8b_instruct \
  --override execution.output_dir=/shared/results
```

**Lepton AI Execution:**
```bash
# With model deployment
nv-eval run --config-dir examples --config-name lepton_nim_llama_3_1_8b_instruct

# Using existing endpoint
nv-eval run --config-dir examples --config-name lepton_none_llama_3_1_8b_instruct
```

## status - Check Job Status

Monitor the status of evaluations.

### Basic Usage

```bash
# Check status of specific invocation
nv-eval status abc12345

# Check status of specific job
nv-eval status abc12345.0
```

### Output Format

The status command returns JSON with job information:

```json
{
  "invocation_id": "abc12345",
  "status": "running",
  "jobs": [
    {
      "job_id": "abc12345.0",
      "task": "hellaswag",
      "status": "completed",
      "progress": "100%"
    }
  ]
}
```

## kill - Kill Jobs

Stop running evaluations.

### Basic Usage

```bash
# Kill entire invocation
nv-eval kill abc12345

# Kill specific job
nv-eval kill abc12345.0
```

### Force Kill

```bash
# Force kill (if supported by executor)
nv-eval kill abc12345 --force
```

## ls - List Resources

List available tasks, runs, and other resources.

### List Tasks

```bash
# List all available evaluation tasks
nv-eval ls tasks

# List with filtering
nv-eval ls tasks --filter language_modeling
```

### List Runs

```bash
# List recent evaluation runs
nv-eval ls runs

# List runs with specific status
nv-eval ls runs --status completed

# List runs from specific time period
nv-eval ls runs --since "2024-01-01"
```

### Output Format

Tasks list includes task name, endpoint type, harness, and container:

```
TASK                ENDPOINT_TYPE    HARNESS          CONTAINER
arc_challenge       chat            lm_eval          nvcr.io/nvidia/nemo:24.01
hellaswag          completions     lm_eval          nvcr.io/nvidia/nemo:24.01
winogrande         completions     lm_eval          nvcr.io/nvidia/nemo:24.01
```

## export - Export Results

Export evaluation results to various destinations.

### Basic Usage

```bash
# Export to local files (JSON format)
nv-eval export abc12345 --dest local --format json

# Export to specific directory
nv-eval export abc12345 --dest local --format json --output-dir ./results
```

### Export Destinations

**Local Files:**
```bash
# JSON format
nv-eval export abc12345 --dest local --format json

# CSV format
nv-eval export abc12345 --dest local --format csv

# Include logs
nv-eval export abc12345 --dest local --format json --copy-logs
```

**MLflow:**
```bash
nv-eval export abc12345 --dest mlflow \
  --tracking-uri http://localhost:5000 \
  --experiment-name "my_evaluation"
```

**Weights & Biases:**
```bash
nv-eval export abc12345 --dest wandb \
  --entity my_org \
  --project evaluations
```

**Google Sheets:**
```bash
nv-eval export abc12345 --dest gsheets \
  --spreadsheet-name "Evaluation Results"
```

### Multiple Invocations

```bash
# Export multiple runs together
nv-eval export abc12345 def67890 ghi11111 --dest local --format json
```

### Configuration File

```bash
# Use configuration file for export settings
nv-eval export abc12345 --dest mlflow --config-file export_config.yaml
```

## version - Version Information

Display version and build information.

```bash
# Show version
nv-eval version

# Alternative
nv-eval --version
```

## Environment Variables

The CLI respects several environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `NEMO_EVAL_CONFIG_DIR` | Default configuration directory | `examples` |
| `NEMO_EVAL_OUTPUT_DIR` | Default output directory | `./results` |
| `NGC_API_KEY` | NVIDIA NGC API key | - |
| `HF_TOKEN` | Hugging Face token | - |

### Usage

```bash
# Set environment variables
export NGC_API_KEY="nvapi-..."
export NEMO_EVAL_OUTPUT_DIR="/shared/results"

# Run evaluation (uses environment variables)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
```

## Configuration File Examples

### Basic Local Configuration

```yaml
# my_configs/basic_eval.yaml
defaults:
  - execution: local
  - deployment: none

execution:
  output_dir: results

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: my-model

evaluation:
  tasks:
    - name: hellaswag
    - name: arc_challenge
```

### Advanced Configuration with Overrides

```yaml
# my_configs/advanced_eval.yaml
defaults:
  - execution: slurm
  - deployment: vllm

execution:
  output_dir: /shared/results
  partition: gpu
  nodes: 1
  gpus_per_node: 8

deployment:
  image: nvcr.io/nvidia/vllm:latest
  model_path: /models/llama-3.1-8b
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    VLLM_USE_FLASH_ATTN: "1"

target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: meta-llama/Llama-3.1-8B-Instruct

evaluation:
  overrides:
    config.params.temperature: 0.7
    config.params.max_new_tokens: 2048
  tasks:
    - name: hellaswag
      overrides:
        config.params.parallelism: 32
    - name: arc_challenge
      env_vars:
        HF_TOKEN: HF_TOKEN
```

## Troubleshooting

### Common Issues

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
# Enable verbose logging
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  --verbose

# Or set log level
export NEMO_EVAL_LOG_LEVEL=DEBUG
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct
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
