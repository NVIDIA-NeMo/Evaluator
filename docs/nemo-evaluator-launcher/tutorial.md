# NeMo Evaluator Launcher Tutorial and Quickstart

Run reproducible evaluations against your own model endpoints. This guide shows the fastest path from a compatible endpoint to first results.

## 1) Install the launcher

```bash
# Install NeMo Evaluator launcher
pip install nemo-evaluator-launcher
```

## 2) Prerequisite: an OpenAI-compatible endpoint

NeMo Evaluator sends OpenAI-compatible requests to your model during evaluation. You must have an endpoint that accepts either chat or completions API calls and can handle the evaluation load.

**Configuration Examples**: Explore ready-to-use configuration files in [`packages/nemo-evaluator-launcher/examples/`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples) for local, Lepton, and Slurm deployments with various model hosting options (vLLM, NIM, hosted endpoints).

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

  See examples for [build.nvidia.com](https://build.nvidia.com/) usage in the [local evaluation tutorial](tutorials/local-evaluation-of-existing-endpoint.md).

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

For more information, see:

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

The `examples/` directory contains ready-to-use configurations:

- Local execution: [examples/local_llama_3_1_8b_instruct.yaml](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/local_llama_3_1_8b_instruct.yaml)
- Slurm execution: [examples/slurm_llama_3_1_8b_instruct.yaml](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/slurm_llama_3_1_8b_instruct.yaml)
- Lepton AI execution: [examples/lepton_nim_llama_3_1_8b_instruct.yaml](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples/lepton_nim_llama_3_1_8b_instruct.yaml)

Run a local evaluation (requires [Docker](https://www.docker.com/)):
```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

See guides for other backends:
- [Slurm](executors/slurm.md)
- [Lepton](executors/lepton.md)

#### Creating Custom Configurations

1. Create your own configuration directory:
```bash
mkdir my_configs
```

2. Copy an example configuration as a starting point:
```bash
cp examples/local_llama_3_1_8b_instruct.yaml my_configs/my_evaluation.yaml
```

3. Modify the [configuration](configuration/index.md) to suit your needs:
   - [Change the model endpoint](configuration/target/index.md)
   - [Adjust evaluation parameters and select different benchmarks](configuration/evaluation/index.md)
   - [Configure deployment settings](configuration/deployment/index.md)
   - [Configure execution settings](configuration/execution/index.md)

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

#### Environment Variables

Environment variables can be specified for all tasks or for specific tasks. In the below example, the `JUDGE_API_KEY_FOR_ALL_TASKS` environment variable is mapped to the `JUDGE_API_KEY` environment variable for all tasks, and the `HF_TOKEN_FOR_GPQA_DIAMOND` environment variable is mapped to the `HF_TOKEN` environment variable for the `gpqa_diamond` task.

```yaml
evaluation:
  env_vars:
    JUDGE_API_KEY: JUDGE_API_KEY_FOR_ALL_TASKS
  tasks:
    - name: AA_AIME_2024
    - name: AA_math_test_500
    - name: gpqa_diamond
      env_vars:
        HF_TOKEN: HF_TOKEN_FOR_GPQA_DIAMOND
```

For [Slurm](executors/slurm.md#environment-variables) and [Lepton](executors/lepton.md#configuration-notes), there are also ways to pass environment variables to the execution (like deployment containers).

### 3. Check Evaluation Status

Monitor the status of your evaluation jobs:

```bash
nemo-evaluator-launcher status <job_id_or_invocation_id>
```

You can check:
- Individual job status: `nemo-evaluator-launcher status <job_id>`
- All jobs in an invocation: `nemo-evaluator-launcher status <invocation_id>`

The status command returns JSON output with job status information.
