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
pip install nemo-evaluator-launcher[mlflow,wandb,gsheets]  # All exporters
pip install nemo-evaluator-launcher[mlflow]               # MLflow only
pip install nemo-evaluator-launcher[wandb]                # W&B only
pip install nemo-evaluator-launcher[gsheets]              # Google Sheets only
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
    -o target.api_endpoint.api_key=${NGC_API_KEY}
  
  # Or using the full command name
  nemo-evaluator-launcher run --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${NGC_API_KEY}
  ```

  See examples for build.nvidia.com usage in the examples/ folder (TODO: link to examples/).


Self-hosted options:

- TRT-LLM:
```bash
trtllm-serve /path/to/your/model \
  --backend pytorch \
  --port 8000
```

- NeMo:
```bash
# TODO: link to NeMo deployment guide
```

- vLLM:
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct
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

**Complete Working Example**: Here's the actual `local_llama_3_1_8b_instruct.yaml` configuration:

```yaml
# examples/local_llama_3_1_8b_instruct.yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: ./results

target:
  api_endpoint:
    url: http://localhost:8080/v1/chat/completions
    model_id: meta/llama-3.1-8b-instruct
    
evaluation:
  tasks:
    - name: hellaswag
      params:
        limit_samples: 100
    - name: arc_challenge  
      params:
        limit_samples: 100
    - name: winogrande
      params:
        limit_samples: 100
```

Run this configuration (requires Docker and a model endpoint):
```bash
# Using short alias (recommended)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>

# Or using full command name  
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

For other backends:
- Slurm: see executors/slurm.md
- Lepton: see executors/lepton.md

#### Lepton Execution Strategy
See executors/lepton.md for Lepton's parallel deployment strategy and examples.


#### Create Custom Configurations

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

#### Environment Variables in Deployment

The platform supports passing environment variables to deployment containers in a Hydra-extensible way:

Direct Values:
```yaml
deployment:
  type: vllm
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    OMP_NUM_THREADS: "1"
    VLLM_USE_FLASH_ATTN: "1"
```

Environment Variable References:
```yaml
deployment:
  type: sglang
  envs:
    HF_TOKEN: ${oc.env:HF_TOKEN}  # References host environment variable
    NGC_API_KEY: ${oc.env:NGC_API_KEY}
```

Supported Executors:
- SLURM: Environment variables are exported in the sbatch script before running deployment commands
- Lepton: Environment variables are passed to the container specification
- Local: Environment variables are passed to Docker containers (when deployment support is added)

Example with SLURM:
```yaml
deployment:
  type: vllm
  envs:
    CUDA_VISIBLE_DEVICES: "0,1,2,3,4,5,6,7"
    HF_TOKEN: ${oc.env:HF_TOKEN}
    VLLM_USE_V2_BLOCK_MANAGER: "1"
  command: vllm serve /checkpoint --port 8000
```

This will generate a sbatch script that exports these variables before running the deployment command.

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
# Limit to 10 samples for testing
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o +config.params.limit_samples=10
```
