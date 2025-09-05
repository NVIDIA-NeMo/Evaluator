(gs-install)=
# Installation Guide

NeMo Evaluator provides multiple installation paths depending on your needs. Choose the approach that best fits your use case.

## Choose Your Installation Path

### 🚀 **NeMo Evaluator Launcher** (Recommended)
**Best for**: Most users who want unified CLI and orchestration across backends

- ✅ Unified CLI for 100+ benchmarks
- ✅ Multi-backend execution (local, Slurm, cloud)
- ✅ Built-in result export to MLflow, W&B, etc.
- ✅ Configuration management with examples

### ⚙️ **NeMo Evaluator Core**  
**Best for**: Developers building custom evaluation pipelines

- ✅ Programmatic Python API
- ✅ Direct container access
- ✅ Custom framework integration
- ✅ Advanced adapter configuration

### 🐳 **Container Direct**
**Best for**: Users who prefer container-based workflows

- ✅ Pre-built NGC evaluation containers
- ✅ Guaranteed reproducibility
- ✅ No local installation required
- ✅ Isolated evaluation environments

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

```bash
# Create and activate virtual environment
python3 -m venv nemo-evaluator-env
source nemo-evaluator-env/bin/activate

# Install NeMo Evaluator Launcher
pip install nemo-evaluator-launcher
```

Quick verification:
```bash
nemo-evaluator-launcher ls tasks
```

:::

:::{tab-item} Core Library

Install NeMo Evaluator Core for programmatic access:

```bash
# Create and activate virtual environment  
python3 -m venv nemo-evaluator-env
source nemo-evaluator-env/bin/activate

# Install core dependencies
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
pip install --no-build-isolation nemo-evaluator

# Install specific evaluation packages as needed
pip install nvidia-simple-evals  # For basic evaluations
pip install nvidia-lm-eval      # For language model benchmarks
```

Quick verification:
```bash
python -c "from nemo_evaluator.core.evaluate import evaluate; print('Core library installed successfully')"
```

:::



:::{tab-item} NGC Containers (Recommended)

Use pre-built evaluation containers from NVIDIA NGC for guaranteed reproducibility:

```bash
# Pull and run a specific evaluation container
docker pull nvcr.io/nvidia/eval-factory/simple-evals:25.07.3
docker run --rm -it --gpus all nvcr.io/nvidia/eval-factory/simple-evals:25.07.3

# Inside container - run evaluations
export MY_API_KEY=your_api_key
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir /tmp/results
```

Available containers:
- `simple-evals` - Basic evaluation tasks
- `lm-evaluation-harness` - Language model benchmarks  
- `bigcode-evaluation-harness` - Code generation
- `safety-harness` - Safety and bias evaluation
- `vlmevalkit` - Vision-language models

See [Container Reference](../nemo-evaluator/reference/containers.md) for complete list.

:::

:::{tab-item} NeMo Framework

For optimal performance with NeMo models, use the NeMo Framework container:

```bash
# Get the latest NeMo Framework container
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}

# Inside container - install NeMo Evaluator
pip install nemo-evaluator-launcher
```

:::

:::{tab-item} UV

To install NeMo Eval with `uv`,  refer to our [Contribution guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md).

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
