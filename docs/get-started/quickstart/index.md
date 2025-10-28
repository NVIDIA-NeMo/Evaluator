(gs-quickstart)=
# Quickstart

Get up and running with NeMo Evaluator in minutes. Choose your preferred approach based on your needs and experience level.

## Prerequisites

All paths require:

- OpenAI-compatible endpoint (hosted or self-deployed)
- Valid API key for your chosen endpoint

## Quick Reference

| Task | Command |
|------|---------|
| List benchmarks | `nemo-evaluator-launcher ls tasks` |
| Run evaluation | `nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name <config>` |
| Check status | `nemo-evaluator-launcher status <invocation_id>` |
| Job info | `nemo-evaluator-launcher info <invocation_id>` |
| Export results | `nemo-evaluator-launcher export <invocation_id> --dest local --format json` |
| Dry run | Add `--dry-run` to any run command |
| Test with limited samples | Add `-o +config.params.limit_samples=3` |


## Choose Your Path

Select the approach that best matches your workflow and technical requirements:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` NeMo Evaluator Launcher
:link: gs-quickstart-launcher
:link-type: ref
**Recommended for most users**

Unified CLI experience with automated container management, built-in orchestration, and result export capabilities.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` NeMo Evaluator Core
:link: gs-quickstart-core
:link-type: ref
**For Python developers**

Programmatic control with full adapter features, custom configurations, and direct API access for integration into existing workflows.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` NeMo Framework Container
:link: gs-quickstart-nemo-fw
:link-type: ref
**For NeMo Framework Users**

End-to-end training and evaluation of large language models (LLMs).
:::

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Container Direct
:link: gs-quickstart-container
:link-type: ref
**For container workflows**

Direct container execution with volume mounting, environment control, and integration into Docker-based CI/CD pipelines.
:::

::::

## Model Endpoints

NeMo Evaluator works with any OpenAI-compatible endpoint. You have several options:

### **Hosted Endpoints** (Recommended)

- **NVIDIA Build**: [build.nvidia.com](https://build.nvidia.com) - Ready-to-use hosted models
- **OpenAI**: Standard OpenAI API endpoints
- **Other providers**: Anthropic, Cohere, or any OpenAI-compatible API

### **Self-Hosted Options**

If you prefer to host your own models, verify OpenAI compatibility using our {ref}`deployment-testing-compatibility` guide.

If you are deploying the model locally with Docker, you can use a dedicated docker network.
This will provide a secure connetion between deployment and evaluation docker containers.

```bash
# create a dedicated docker network
docker network create my-custom-network

# launch deployment
docker run --gpus all --network my-custom-network --name my-phi-container vllm/vllm-openai:latest \
    --model microsoft/Phi-4-mini-instruct --max-model-len 8192

# Or use other serving frameworks
# TRT-LLM, NeMo Framework, etc.
```

Create an evaluation config:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: my_phi_test
  extra_docker_args: "--network my-custom-network"  # same network as used for deployment

target:
  api_endpoint:
    model_id: microsoft/Phi-4-mini-instruct
    url: http://my-phi-container:8000/v1/chat/completions
    api_key_name: null

evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
      nemo_evaluator_config:
        config:
          params:
            limit_samples: 10 # TEST ONLY: Limits to 10 samples for quick testing
            parallelism: 1
```

Save the config to a file (e.g. `phi-eval.yaml`) and launch the evaluation:

```bash
nemo-evaluator-launcher run \
    --config-dir . \
    --config-name phi-eval.yaml \
    -o execution.output_dir=./phi-results


<!-- See {ref}`deployment-overview` for detailed deployment options. -->

## Validation and Troubleshooting

### Quick Validation Steps

Before running full evaluations, verify your setup:

```bash
# 1. Test your endpoint connectivity
export NGC_API_KEY=nvapi-...
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
    -H "Authorization: Bearer $NGC_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 10
    }'

# 2. Run a dry-run to validate configuration
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run

# 3. Run a minimal test with very few samples
nemo-evaluator-launcher run \
    --config-dir packages/nemo-evaluator-launcher/examples \
    --config-name local_llama_3_1_8b_instruct \
    -o +config.params.limit_samples=1 \
    -o execution.output_dir=./test_results
```

### Common Issues and Solutions

::::{tab-set}

:::{tab-item} API Key Issues
:sync: api-key

```bash
# Verify your API key is set correctly
echo $NGC_API_KEY

# Test with a simple curl request (see above)
```
:::

:::{tab-item} Container Issues
:sync: container

```bash
# Check Docker is running and has GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base-ubuntu20.04 nvidia-smi

# Pull the latest container if you have issues
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
```
:::

:::{tab-item} Configuration Issues
:sync: config

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check available evaluation types
nemo-evaluator-launcher ls tasks
```
:::

:::{tab-item} Result Validation
:sync: results

```bash
# Check if results were generated
find ./results -name "*.yml" -type f

# View task results
cat ./results/<invocation_id>/<task_name>/artifacts/results.yml

# Or export and view processed results
nemo-evaluator-launcher export <invocation_id> --dest local --format json
cat ./results/<invocation_id>/processed_results.json
```
:::

::::

## Next Steps

After completing your quickstart:

::::{tab-set}

:::{tab-item} Explore More Benchmarks
:sync: benchmarks

```bash
# List all available tasks
nemo-evaluator-launcher ls tasks

# Run with limited samples for quick testing
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name local_limit_samples
```
:::

:::{tab-item} Export Results
:sync: export

```bash
# Export to MLflow
nemo-evaluator-launcher export <invocation_id> --dest mlflow

# Export to Weights & Biases
nemo-evaluator-launcher export <invocation_id> --dest wandb

# Export to Google Sheets
nemo-evaluator-launcher export <invocation_id> --dest gsheets

# Export to local files
nemo-evaluator-launcher export <invocation_id> --dest local --format json
```
:::

:::{tab-item} Scale to Clusters
:sync: scale

```bash
cd packages/nemo-evaluator-launcher
# Run on Slurm cluster
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name slurm_llama_3_1_8b_instruct

# Run on Lepton AI
nemo-evaluator-launcher run --config-dir packages/nemo-evaluator-launcher/examples --config-name lepton_vllm_llama_3_1_8b_instruct
```
:::

::::


```{toctree}
:maxdepth: 1
:hidden:

NeMo Evaluator Launcher <launcher>
NeMo Evaluator Core <core>
NeMo Framework Container <nemo-fw>
Container Direct <container>
```
