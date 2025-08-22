# Install Eval

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU(s) (tested on RTX A6000, A100, H100)
- NeMo Framework container (recommended)

### Requirements

- Python 3.12
- PyTorch 2.7
- CUDA 12.9
- Ubuntu 24.04

---

## Installation Methods

### Use pip

For quick exploration of NeMo Eval, we recommend installing our pip package:

```bash
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
pip install --no-build-isolation nemo-eval
```

### Use Docker

For optimal performance and user experience, use the latest version of the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent $TAG and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

### Use uv

To install NeMo Eval with uv, please refer to our [Contribution guide](https://github.com/NVIDIA-NeMo/Eval/blob/main/CONTRIBUTING.md).

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
