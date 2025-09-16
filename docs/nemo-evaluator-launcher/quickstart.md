# NeMo Evaluator Launcher Tutorial and Quickstart

Run reproducible evaluations against your own model endpoints. This guide shows the fastest path from a compatible endpoint to first results.

## 1) Install the launcher

```bash
# Install NeMo Evaluator launcher
pip install nemo-evaluator-launcher
```

## 2) Prerequisite: an OpenAI-compatible endpoint

NeMo Evaluator sends OpenAI-compatible requests to your model during evaluation. You must have an endpoint that accepts either chat or completions API calls and can handle the evaluation load.

Hosted endpoints (fastest):

- [build.nvidia.com](https://build.nvidia.com) (ready-to-use hosted models):
  Hosted models expose OpenAI‑compatible APIs and work out of the box for evaluations — no hosting required. This is the fastest, least‑effort path to run evals across available endpoints.

  Example model: [nvidia/llama-3.1-nemotron-nano-vl-8b-v1](https://build.nvidia.com/nvidia/llama-3.1-nemotron-nano-vl-8b-v1)

  Minimal usage (override endpoint URL and key):
  ```bash
  nemo-evaluator-launcher run --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    -o target.api_endpoint.url=https://integrate.api.nvidia.com/v1/chat/completions \
    -o target.api_endpoint.api_key=${NGC_API_KEY}
  ```

  For NVIDIA APIs, see [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key).

  See examples for [build.nvidia.com](https://build.nvidia.com/) usage in the examples/ folder (TODO: link to examples/).

Self-hosted options:

- TRT-LLM:
```bash
trtllm-serve /path/to/your/model \
  --backend pytorch \
  --port 8000
```

- vLLM:
```bash
docker run --gpus all -p 8000:8000 vllm/vllm-openai:latest \
  --model meta-llama/Llama-3.1-8B-Instruct
```

Optional: quick endpoint check
```bash
nemo-evaluator-launcher test-endpoint --url http://localhost:8000/v1
```
  **Self-hosted options:**

  For detailed deployment instructions, see the [Deployment Frameworks Guide](tutorials/deployments/deployment-frameworks-guide.md).

  To test your endpoint compatibility, see [Testing Endpoint Compatibility](tutorials/deployments/testing-endpoint-oai-compatibility.md).

## Quick Start

### 1. List Available Benchmarks

View all available evaluation benchmarks:

```bash
nemo-evaluator-launcher ls tasks
```

### 2. Run Evaluations

The NeMo Evaluator Launcher uses [Hydra](https://hydra.cc/docs/intro/) for configuration management. You can run evaluations using predefined configurations or create your own.

#### Using Example Configurations

The examples/ directory contains ready-to-use configurations:

- Local execution: examples/local_llama_3_1_8b_instruct.yaml
- Slurm execution: see executors guide (executors/slurm.md)
- Lepton AI execution: see executors guide (executors/lepton.md)

Run a local evaluation (requires [Docker](https://www.docker.com/)):
```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

For other backends:
- Slurm: see executors/slurm.md
- Lepton: see executors/lepton.md

#### Lepton Execution Strategy
See executors/lepton.md for Lepton’s parallel deployment strategy and examples.


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
nemo-evaluator-launcher run --config-dir my_configs --config-name my_evaluation
```

#### Configuration Overrides

You can override configuration values from the command line (`-o` can be used multiple times, the notation follows [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/)):

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=my_model
```

#### Environment Variables in Deployment

The platform supports passing environment variables to deployment containers in a [Hydra](https://hydra.cc/docs/intro/)-extensible way:

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
- Local: Environment variables are passed to [Docker](https://www.docker.com/) containers (when deployment support is added)

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
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

You can check:
- Individual job status: `nemo-evaluator-launcher status <job_id>`
- All jobs in an invocation: `nemo-evaluator-launcher status <invocation_id>`

The status command returns JSON output with job status information.
