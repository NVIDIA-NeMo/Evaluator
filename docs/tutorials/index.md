(tutorials-overview)=

# Tutorials

Master NeMo Eval with hands-on tutorials and practical examples.

## Before You Start

Before starting the tutorials, ensure you have:

- **NeMo Framework Container**: Running the latest [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo)
- **Model Checkpoint**: Access to a NeMo 2.0 checkpoint (tutorials use Llama 3.2 1B Instruct)
- **GPU Resources**: CUDA-compatible GPU with sufficient memory
- **Jupyter Environment**: Ability to run Jupyter notebooks

---

## Available Tutorials

Build your expertise with these progressive tutorials:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`play;1.5em;sd-mr-1` 1. MMLU Evaluation
:link: https://github.com/NVIDIA-NeMo/Eval/tree/main/tutorials/mmlu.ipynb
:link-type: url
Deploy models and run evaluations with the MMLU benchmark for both completions and chat endpoints.
:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` 2. Simple Evals Framework
:link: https://github.com/NVIDIA-NeMo/Eval/tree/main/tutorials/simple-evals.ipynb
:link-type: url
Discover how to extend evaluation capabilities by installing additional frameworks and running HumanEval coding assessments.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` 3. Custom Tasks
:link: https://github.com/NVIDIA-NeMo/Eval/tree/main/tutorials/wikitext.ipynb
:link-type: url
Master custom evaluation workflows by running WikiText benchmark with advanced configuration and log-probability analysis.
:::

::::

## Run the Tutorials

1. Start NeMo Framework Container:
   ```bash
   docker run --rm -it -w /workdir -v $(pwd):/workdir \
     --entrypoint bash --gpus all \
     nvcr.io/nvidia/nemo:${TAG}
   ```

2. Launch Jupyter:
   ```bash
   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
   ```

3. Navigate to the `tutorials/` directory and open the desired notebook
