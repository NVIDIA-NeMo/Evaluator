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
- Python 3.10 or higher
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

TODO

:::

:::{tab-item} Core Library

Install NeMo Evaluator Core for programmatic access:

TODO

Quick verification:
```bash
python -c "from nemo_evaluator.core.evaluate import evaluate; print('Core library installed successfully')"
```

:::

:::{tab-item} NGC Containers

Use pre-built evaluation containers from NVIDIA NGC for guaranteed reproducibility:

TODO

:::

::::

---

(optional-packages)=

## Add New Evaluation Frameworks
The NeMo Framework Docker image comes with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/) pre-installed.
However, you can add more evaluation methods by installing additional NVIDIA Eval Factory packages.

For each package, follow these steps:

1. Install the required package.

2. Deploy your model:

```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:start-after: "## Deploy"
:linenos:
```

Run the deployment in the background:

```bash
python deploy.py
```

Make sure to open two separate terminals within the same container for executing the deployment and evaluation.

3. (Optional) Export the required environment variables. 

4. Run the evalution of your choice.

Below you can find examples for enabling and launching evaluations for different packages.
Note that all example use only a subset of samples.
To run the evaluation on the whole dataset, remove the `"limit_samples"` parameter.

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
export JUDGE_API_KEY=...
```

To customize the judge setting, see the instructions for [NVIDIA Eval Factory package](https://pypi.org/project/nvidia-simple-evals/). 

3. Run the evaluation:

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

   If you set a gated judge endpoint up, you must export your API key as the `JUDGE_API_KEY` variable:

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
