(launcher-quickstart)=

# NeMo Evaluator Launcher Quickstart

Run reproducible evaluations against your own model endpoints. This guide shows the fastest path from a compatible endpoint to first results.

## 1) Install the launcher

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install NeMo Evaluator launcher
pip install nemo-evaluator-launcher

# Optional: Install with specific exporters
pip install "nemo-evaluator-launcher[mlflow,wandb,gsheets]"  # All exporters
pip install "nemo-evaluator-launcher[mlflow]"               # MLflow only
pip install "nemo-evaluator-launcher[wandb]"                # W&B only
pip install "nemo-evaluator-launcher[gsheets]"              # Google Sheets only
```

**Requirements:**

- Python 3.10 to 3.13
- Docker (for local execution)
- Access to model endpoints (hosted or self-deployed)

## 2) Prerequisite: an OpenAI-compatible endpoint

NeMo Evaluator sends OpenAI-compatible requests to your model during evaluation. You must have an endpoint that accepts either chat or completions API calls and can handle the evaluation load.

Hosted endpoints (fastest):

- build.nvidia.com (ready-to-use hosted models):
  Hosted models expose OpenAI‑compatible APIs and work out of the box for evaluations — no hosting required. This is the fastest, least‑effort path to run evals across available endpoints.

  Example model: [nvidia/llama-3.1-nemotron-nano-vl-8b-v1](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1)

  Minimal usage (override endpoint URL and key):

  ```bash
  # Using the short alias (recommended)
  nv-eval run --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key_name=NGC_API_KEY
  
  # Or using the full command name
  nemo-evaluator-launcher run --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key_name=NGC_API_KEY
  ```

## Quick Start

### 1. List Available Benchmarks

View all available evaluation benchmarks:

```bash
# List all available tasks/benchmarks
nv-eval ls tasks

# Alternative: list recent runs
nv-eval ls runs
```

### 2. Run Evaluations

The NeMo Evaluator Launcher uses Hydra for configuration management. You can run evaluations using predefined configurations or create your own.

#### Use Example Configurations

The examples/ directory contains ready-to-use configurations:

- Local execution: `examples/local_llama_3_1_8b_instruct.yaml`
- Slurm execution: `examples/slurm_llama_3_1_8b_instruct.yaml`  
- Lepton AI execution: `examples/lepton_vllm_llama_3_1_8b_instruct.yaml`

**Complete Working Example**: Here's an excerpt from the actual `local_llama_3_1_8b_instruct.yaml` configuration:

```yaml
# examples/local_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: llama_3_1_8b_instruct_results

target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY
    
evaluation:
  overrides:
    config.params.request_timeout: 3600
    target.api_endpoint.adapter_config.use_reasoning: false
  tasks:
    - name: ifeval
    - name: gpqa_diamond
      overrides:
        config.params.temperature: 0.6
        config.params.top_p: 0.95
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
    - name: mbpp
      overrides:
        config.params.temperature: 0.2
```

Run this configuration (requires Docker and a model endpoint):

```bash
# Using short alias (recommended)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>

# Or using full command name  
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

For other backends:

- SLURM: see [SLURM executor configuration](configuration/executors/slurm.md)
- Lepton: see [Lepton executor configuration](configuration/executors/lepton.md)

#### Create Custom Configurations

1. Create your own configuration directory:

   ```bash
   mkdir my_configs
   ```

2. Copy an example configuration as a starting point:

   ```bash
   cp examples/local_llama_3_1_8b_instruct.yaml my_configs/my_evaluation.yaml
   ```

3. Edit the configuration to suit your needs:

   - Change the model endpoint
   - Adjust evaluation parameters
   - Select different benchmarks
   - Configure execution settings

4. Run your custom configuration:

   ```bash
   # Using short alias
   nv-eval run --config-dir my_configs --config-name my_evaluation

   # Or using full command
   nemo-evaluator-launcher run --config-dir my_configs --config-name my_evaluation
   ```

#### Configuration Overrides

You can override configuration values from the command line (`-o` can be used multiple times, the notation follows Hydra):

```bash
# Using short alias (recommended)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one

# Or using full command
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

### 3. Check Evaluation Status

Monitor the status of your evaluation jobs:

```bash
# Check status using short alias
nv-eval status <job_id_or_invocation_id>

# Or using full command
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

You can check:

- Individual job status: `nv-eval status <job_id>`
- All jobs in an invocation: `nv-eval status <invocation_id>`
- Kill running jobs: `nv-eval kill <job_id_or_invocation_id>`

The status command returns JSON output with job status information.

/// note | About invocation and job IDs
It is possible to use short version of IDs in `status` command, for example `abcd` instead of a full `abcdef0123456` or `ab.0` instead of `abcdef0123456.0`, so long as there are no collisions. This is a syntactic sugar allowing for a slightly easier usage.
///

### 4. Export Results

Export evaluation results to various destinations:

```bash
# Export to local files (JSON/CSV)
nv-eval export <invocation_id> --dest local --format json

# Export to MLflow
nv-eval export <invocation_id> --dest mlflow

# Export to Weights & Biases
nv-eval export <invocation_id> --dest wandb

# Export to Google Sheets
nv-eval export <invocation_id> --dest gsheets
```

### 5. Troubleshooting

View the full resolved configuration without running:

```bash
# Dry run to see full config
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct --dry-run
```

Test a small subset before running full benchmarks:

```bash
# Add global override to limit all tasks to 10 samples for testing
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +evaluation.overrides.config.params.limit_samples=10
```
