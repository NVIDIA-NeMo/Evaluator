(gs-quickstart)=
# Quickstart

Get up and running with NeMo Evaluator in minutes. Choose your preferred approach based on your needs and experience level.

## Prerequisites

All paths require:

- OpenAI-compatible endpoint (hosted or self-deployed)
- Valid API key for your chosen endpoint

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

:::{grid-item-card} {octicon}`container;1.5em;sd-mr-1` Container Direct
:link: gs-quickstart-container
:link-type: ref
**For container workflows**

Direct container execution with volume mounting, environment control, and integration into Docker-based CI/CD pipelines.
:::

:::{grid-item-card} {octicon}`stack;1.5em;sd-mr-1` Complete Stack Integration
:link: gs-quickstart-full-stack
:link-type: ref
**For advanced users**

Full three-tier architecture combining model deployment, advanced evaluation, and orchestration for production workflows.
:::

::::

## Model Endpoints

NeMo Evaluator works with any OpenAI-compatible endpoint. You have several options:

### **Hosted Endpoints** (Recommended)

- **NVIDIA Build**: [build.nvidia.com](https://build.nvidia.com) - Ready-to-use hosted models
- **OpenAI**: Standard OpenAI API endpoints  
- **Other providers**: Anthropic, Cohere, or any OpenAI-compatible API

### **Self-Hosted Options**

If you prefer to host your own models:

```bash
# vLLM (recommended for self-hosting)
pip install vllm
vllm serve meta-llama/Llama-3.1-8B-Instruct --port 8080

# Or use other serving frameworks
# TRT-LLM, NeMo Framework, Ray Serve, etc.
```

See {ref}`deployment-overview` for detailed deployment options.

## Validation and Troubleshooting

### Quick Validation Steps

Before running full evaluations, verify your setup:

```bash
# 1. Test your endpoint connectivity
curl -X POST "https://integrate.api.nvidia.com/v1/chat/completions" \
    -H "Authorization: Bearer $NGC_API_KEY" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta/llama-3.1-8b-instruct",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 10
    }'

# 2. Run a dry-run to validate configuration
nv-eval run \
    --config-dir examples \
    --config-name local_llama_3_1_8b_instruct \
    --dry-run

# 3. Run a minimal test with very few samples
nv-eval run \
    --config-dir examples \
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
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Check available evaluation types
nv-eval ls tasks
```
:::

:::{tab-item} Result Validation
:sync: results

```bash
# Check if results were generated
find ./results -name "*.json" -type f

# View summary results
cat ./results/<invocation_id>/summary.json
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
nv-eval ls tasks

# Run different evaluation types
nv-eval run --config-dir examples --config-name local_safety_evaluation
nv-eval run --config-dir examples --config-name local_code_generation
```
:::

:::{tab-item} Export Results
:sync: export

```bash
# Export to MLflow
nv-eval export <invocation_id> --dest mlflow

# Export to Weights & Biases  
nv-eval export <invocation_id> --dest wandb

# Export to local files
nv-eval export <invocation_id> --dest local --format json
```
:::

:::{tab-item} Scale to Clusters
:sync: scale

```bash
# Run on Slurm cluster
nv-eval run --config-dir examples --config-name slurm_multi_gpu

# Run on Lepton AI
nv-eval run --config-dir examples --config-name lepton_deployment
```
:::

::::

### Quick Reference

| Task | Command |
|------|---------|
| List benchmarks | `nv-eval ls tasks` |
| Run evaluation | `nv-eval run --config-dir examples --config-name <config>` |
| Check status | `nv-eval status <invocation_id>` |
| Export results | `nv-eval export <invocation_id> --dest local --format json` |
| Dry run | Add `--dry-run` to any run command |
| Test with limited samples | Add `-o +config.params.limit_samples=3` |

```{toctree}
:maxdepth: 1
:hidden:

NeMo Evaluator Launcher <launcher>
NeMo Evaluator Core <core>
Container Direct <container>
Complete Stack Integration <full-stack>
```
