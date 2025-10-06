(gs-install)=
# Installation Guide

NeMo Evaluator provides multiple installation paths depending on your needs. Choose the approach that best fits your use case.

## Choose Your Installation Path

```{list-table} Installation Path Comparison
:header-rows: 1
:widths: 25 25 50

* - **Installation Path**
  - **Best For**
  - **Key Features**
* - **NeMo Evaluator Launcher** (Recommended)
  - Most users who want unified CLI and orchestration across backends
  - • Unified CLI for 100+ benchmarks  
    • Multi-backend execution (local, Slurm, cloud)  
    • Built-in result export to MLflow, W&B, etc.  
    • Configuration management with examples
* - **NeMo Evaluator Core**
  - Developers building custom evaluation pipelines
  - • Programmatic Python API  
    • Direct container access  
    • Custom framework integration  
    • Advanced adapter configuration
* - **Container Direct**
  - Users who prefer container-based workflows
  - • Pre-built NGC evaluation containers  
    • Guaranteed reproducibility  
    • No local installation required  
    • Isolated evaluation environments
```

---

## Prerequisites

### System Requirements

- Python 3.10 or higher (supports 3.10, 3.11, 3.12, and 3.13)
- CUDA-compatible GPU(s) (tested on RTX A6000, A100, H100)
- Docker (for container-based workflows)

### Recommended Environment

- Python 3.12
- PyTorch 2.7
- CUDA 12.9
- Ubuntu 24.04

---

## Installation Methods

### Use pip

::::{tab-set}

:::{tab-item} Launcher (Recommended)

Install NeMo Evaluator Launcher for unified CLI and orchestration:

```{literalinclude} _snippets/install_launcher.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

Quick verification:

```{literalinclude} _snippets/verify_launcher.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::

:::{tab-item} Core Library

Install NeMo Evaluator Core for programmatic access:

```{literalinclude} _snippets/install_core.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

Quick verification:

```{literalinclude} _snippets/verify_core.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::

:::{tab-item} NGC Containers

Use pre-built evaluation containers from NVIDIA NGC for guaranteed reproducibility:

```bash
# Pull evaluation containers (no local installation needed)
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:{{ docker_compose_latest }}
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
```

```bash
# Run container interactively
docker run --rm -it --gpus all \
    -v $(pwd)/results:/workspace/results \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} bash

# Or run evaluation directly
docker run --rm --gpus all \
    -v $(pwd)/results:/workspace/results \
    -e MY_API_KEY=your-api-key \
    nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
    eval-factory run_eval \
        --eval_type mmlu_pro \
        --model_url https://integrate.api.nvidia.com/v1/chat/completions \
        --model_id meta/llama-3.1-8b-instruct \
        --api_key_name MY_API_KEY \
        --output_dir /workspace/results
```

Quick verification:

```bash
# Test container access
docker run --rm nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }} \
    eval-factory ls | head -5
echo " Container access verified"
```

:::

::::

---

(optional-packages)=

## Add New Evaluation Frameworks

You can add more evaluation methods by installing additional NVIDIA Eval Factory packages.

**Prerequisites**: An OpenAI-compatible model endpoint must be running and accessible.

For each package:

1. Install the required package.

2. Export any required environment variables (if specified).

3. Run the evaluation of your choice.

Below you can find examples for enabling and launching evaluations for different packages.
These examples demonstrate functionality using a subset of samples.
To run the evaluation on the entire dataset, remove the `"limit_samples"` parameter.

::::{tab-set}

:::{tab-item} BFCL

1. Install the [nvidia-bfcl](https://pypi.org/project/nvidia-bfcl/) package:

   ```bash
   pip install nvidia-bfcl==25.7.1
   ```

2. Run the evaluation:

   ```{literalinclude} ../scripts/snippets/bfcl.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

:::

:::{tab-item} garak

1. Install the [nvidia-eval-factory-garak](https://pypi.org/project/nvidia-eval-factory-garak/) package:

   ```bash
   pip install nvidia-eval-factory-garak==25.6
   ```

2. Run the evaluation:

   ```{literalinclude} ../scripts/snippets/garak.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

:::

:::{tab-item} BigCode

1. Install the [nvidia-bigcode-eval](https://pypi.org/project/nvidia-bigcode-eval/) package:

   ```bash
   pip install nvidia-bigcode-eval==25.6
   ```

2. Run the evaluation:

   ```{literalinclude} ../scripts/snippets/bigcode.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

:::

:::{tab-item} simple-evals

1. Install the [nvidia-simple-evals](https://pypi.org/project/nvidia-simple-evals/) package:

   ```bash
   pip install nvidia-simple-evals==25.7.1
   ```

In the example below, we use the `AIME_2025` task, which follows the llm-as-a-judge approach for checking the output correctness.
By default, [Llama 3.3 70B](https://build.nvidia.com/meta/llama-3_3-70b-instruct) NVIDIA NIM is used for judging.

1. To run evaluation, set your [build.nvidia.com](https://build.nvidia.com/) API key as the `JUDGE_API_KEY` variable:

   ```bash
   export JUDGE_API_KEY=your-api-key-here
   ```

To customize the judge setting, see the instructions for [NVIDIA Eval Factory package](https://pypi.org/project/nvidia-simple-evals/). 

1. Run the evaluation:

   ```{literalinclude} ../scripts/snippets/simple_evals.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

:::

:::{tab-item} safety-harness

1. Install the [nvidia-safety-harness](https://pypi.org/project/nvidia-safety-harness/) package:

   ```bash
   pip install nvidia-safety-harness==25.6
   ```

2. Deploy the judge model

   In the example below, we use the `aegis_v2` task, which requires the [Llama 3.1 NemoGuard 8B ContentSafety](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html) model to assess your model's responses.

   The model is available through NVIDIA NIM.
   See the [instructions](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html) on deploying the judge model.

   If you set a gated judge endpoint up, you must export your API key as the ``JUDGE_API_KEY`` variable:

   ```bash
   export JUDGE_API_KEY=...
   ```

3. To access the evaluation dataset, you must authenticate with the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

4. Run the evaluation:

   ```{literalinclude} ../scripts/snippets/safety.py
   :language: python
   :start-after: "## Run the evaluation"
   :linenos:
   ```

   Make sure to modify the judge configuration in the provided snippet to match your Llama 3.1 NemoGuard 8B ContentSafety endpoint:

   ```python
       params={
           "extra": {
               "judge": {
                   "model_id": "my-llama-3.1-nemoguard-8b-content-safety-endpoint",
                   "url": "http://my-hostname:1234/v1/completions",
               }
           }
       }
   ```

:::

::::
