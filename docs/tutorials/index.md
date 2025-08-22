(tutorials-overview)=

# Tutorials

Learn NeMo Eval through hands-on tutorials and practical examples.

## Overview

These step-by-step tutorials guide you through the core workflows of NeMo Eval, from basic model deployment to advanced evaluation techniques. Each tutorial is designed to be self-contained and provides practical examples you can follow along.

## Available Tutorials

The tutorials are organized to build your knowledge progressively:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` MMLU Evaluation
:link: ../tutorials/mmlu.ipynb
:link-type: url
Learn the basics of deploying models and running evaluations using the MMLU benchmark for both completions and chat endpoints.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Simple Evals Framework
:link: ../tutorials/simple-evals.ipynb
:link-type: url
Discover how to extend evaluation capabilities by installing additional frameworks and running HumanEval coding assessments.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Custom Tasks
:link: ../tutorials/wikitext.ipynb
:link-type: url
Master custom evaluation workflows by running WikiText benchmark with advanced configuration and log-probability analysis.
:::

::::

## Tutorial Content

### 1. **MMLU Evaluation** (`mmlu.ipynb`)
- **Focus**: Basic deployment and evaluation workflows
- **Topics Covered**:
  - Model deployment with PyTriton backend
  - Understanding completions vs. chat endpoints
  - Running evaluations on academic benchmarks
  - Interpreting evaluation results and artifacts
- **Best For**: First-time users getting familiar with NeMo Eval

### 2. **Simple Evals Framework** (`simple-evals.ipynb`)
- **Focus**: Extending evaluation capabilities
- **Topics Covered**:
  - Installing additional evaluation packages
  - Working with multiple evaluation harnesses
  - Code generation evaluation with HumanEval
  - Managing framework conflicts and task naming
- **Best For**: Users wanting to expand beyond core benchmarks

### 3. **Custom Tasks** (`wikitext.ipynb`)
- **Focus**: Advanced evaluation configuration
- **Topics Covered**:
  - Defining custom evaluation tasks
  - Working with log-probability evaluations
  - Tokenizer configuration requirements
  - Advanced parameter configuration
- **Best For**: Advanced users with specific evaluation needs

## Prerequisites

Before starting the tutorials, ensure you have:

- **NeMo Framework Container**: Running the latest [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- **Model Checkpoint**: Access to a NeMo 2.0 checkpoint (tutorials use Llama 3.2 1B Instruct)
- **GPU Resources**: CUDA-compatible GPU with sufficient memory
- **Jupyter Environment**: Ability to run Jupyter notebooks

## Running the Tutorials

1. **Start NeMo Framework Container**:
   ```bash
   docker run --rm -it -w /workdir -v $(pwd):/workdir \
     --entrypoint bash --gpus all \
     nvcr.io/nvidia/nemo:${TAG}
   ```

2. **Launch Jupyter**:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
   ```

3. **Open Tutorial**: Navigate to the `tutorials/` directory and open the desired notebook

## Tutorial Structure

Each tutorial follows a consistent structure:

- **Introduction**: Overview and learning objectives  
- **Setup**: Environment preparation and checkpoint configuration
- **Deployment**: Model deployment with specific backend
- **Evaluation**: Step-by-step evaluation execution
- **Results**: Interpreting outputs and next steps
- **Cleanup**: Proper resource cleanup

## Advanced Usage

For advanced users, the tutorials also demonstrate:

- **Multi-GPU Deployment**: Configuring tensor and pipeline parallelism
- **Custom Configurations**: Modifying evaluation parameters
- **Result Analysis**: Understanding metrics and output artifacts
- **Troubleshooting**: Common issues and debugging techniques

## Next Steps

After completing the tutorials:

- Explore the [Evaluation Guide](../evaluation/index.md) for detailed reference material
- Learn about [Deployment Options](../deployment/index.md) for production scenarios
- Review [Advanced Features](../features/index.md) for specialized use cases

## Support

If you encounter issues with the tutorials:

- **Documentation**: Check the relevant user guides for detailed explanations
- **GitHub Issues**: Report tutorial-specific problems
- **Community**: Ask questions in GitHub Discussions