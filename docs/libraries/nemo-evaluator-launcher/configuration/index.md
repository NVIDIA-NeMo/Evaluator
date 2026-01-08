(configuration-overview)=

# Configuration

The nemo-evaluator-launcher uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, enabling flexible composition and command-line overrides.

## How it Works

1. **Choose your deployment**: Start with `deployment: none` to use existing endpoints
2. **Set your execution platform**: Use `execution: local` for development
3. **Configure your target**: Point to your API endpoint
4. **Select benchmarks**: Add evaluation tasks
5. **Test first**: Always use `--dry-run` to verify

```bash
# Verify configuration
nemo-evaluator-launcher run --config your_config.yaml --dry-run

# Run evaluation
nemo-evaluator-launcher run --config your_config.yaml
```

### Basic Structure

Every configuration has four main sections:

```yaml
defaults:
  - execution: local     # Where to run: local, lepton, slurm
  - deployment: none     # How to deploy: none, vllm, sglang, nim, trtllm, generic
  - _self_

execution:
  output_dir: results    # Required: where to save results

target:                  # Required for deployment: none
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY

evaluation:              # Required: what benchmarks to run
  - name: gpqa_diamond
  - name: ifeval
```

## Deployment Options

Choose how to serve your model for evaluation:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`globe;1.5em;sd-mr-1` None (External)
:link: deployment/none
:link-type: doc

Use existing API endpoints like NVIDIA API Catalog, OpenAI, or custom deployments. No model deployment needed.
:::

:::{grid-item-card} {octicon}`broadcast;1.5em;sd-mr-1` vLLM
:link: deployment/vllm
:link-type: doc

High-performance LLM serving with advanced parallelism strategies. Best for production workloads and large models.
:::

:::{grid-item-card} {octicon}`zap;1.5em;sd-mr-1` SGLang
:link: deployment/sglang
:link-type: doc

Fast serving framework optimized for structured generation and high-throughput inference with efficient memory usage.
:::

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` NIM
:link: deployment/nim
:link-type: doc

NVIDIA-optimized inference microservices with automatic scaling, optimization, and enterprise-grade features.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` TRT-LLM
:link: deployment/trtllm
:link-type: doc


NVIDIA TensorRT LLM.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Generic
:link: deployment/generic
:link-type: doc


Deploy models using a fully custom setup.
:::

::::

## Execution Platforms

Choose where to run your evaluations:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`desktop-download;1.5em;sd-mr-1` Local
:link: executors/local
:link-type: doc

Docker-based evaluation on your local machine. Perfect for development, testing, and small-scale evaluations.
:::

:::{grid-item-card} {octicon}`cloud;1.5em;sd-mr-1` Lepton
:link: executors/lepton
:link-type: doc

Cloud execution with on-demand GPU provisioning. Ideal for production evaluations and scalable workloads.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` SLURM
:link: executors/slurm
:link-type: doc

HPC cluster execution with resource management. Best for large-scale evaluations and batch processing.
:::

::::

## Evaluation Configuration

::::{grid} 1 1 1 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Tasks & Benchmarks
:link: evaluation/index
:link-type: doc

Configure evaluation tasks, parameter overrides, and environment variables for your benchmarks.
:::

::::

## Command Line Overrides

Override any configuration value using the `-o` flag:

```bash
# Basic override
nemo-evaluator-launcher run --config your_config.yaml \
  -o execution.output_dir=my_results

# Multiple overrides
nemo-evaluator-launcher run --config your_config.yaml \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.url="https://new-endpoint.com/v1/chat/completions"
```

```{toctree}
:caption: Configuration
:hidden:

Deployment <deployment/index>
Executors <executors/index>
Evaluation <evaluation/index>
```
