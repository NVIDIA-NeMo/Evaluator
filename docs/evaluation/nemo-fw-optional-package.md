# Add Evaluation Packages to NeMo Framework

The NeMo Framework Docker image comes with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/) pre-installed.
However, you can add more evaluation methods by installing additional NeMo Evaluator packages.

For each package, follow these steps:

1. Install the required package.

2. Deploy your model:

```{literalinclude} ../get-started/_snippets/deploy.sh
:language: shell
:start-after: "## Deploy"
```

Wait for the server to get started and ready for accepting requests:
```python
from nemo_evaluator.api import check_endpoint
check_endpoint(
    endpoint_url="http://0.0.0.0:8080/v1/completions/",
    endpoint_type="completions",
    model_name="megatron_model",
)
```

Make sure to open two separate terminals within the same container for executing the deployment and evaluation.

3. (Optional) Export the required environment variables. 

4. Run the evalution of your choice.

Below you can find examples for enabling and launching evaluations for different packages.

:::{tip}
All examples below use only a subset of samples.
To run the evaluation on the whole dataset, remove the `limit_samples` parameter.
:::

## Enable On-Demand Evaluation Packages

:::{note}
If multiple harnesses are installed in your environment and they define a task with the same name, you must use the `<harness>.<task>` format to avoid ambiguity. For example:

```python
eval_config = EvaluationConfig(type="lm-evaluation-harness.mmlu")
eval_config = EvaluationConfig(type="simple_evals.mmlu")
```
:::

::::{tab-set}

:::{tab-item} BFCL

1. Install the [nvidia-bfcl](https://pypi.org/project/nvidia-bfcl/) package:

```bash
pip install nvidia-bfcl
```

2. Run the evaluation:

```{literalinclude} _snippets/bfcl.py
:language: python
:start-after: "## Run the evaluation"
```
:::

:::{tab-item} garak

1. Install the [nvidia-eval-factory-garak](https://pypi.org/project/nvidia-eval-factory-garak/) package:

```bash
pip install nvidia-eval-factory-garak
```

2. Run the evaluation:

```{literalinclude} _snippets/garak.py
:language: python
:start-after: "## Run the evaluation"
```

:::

:::{tab-item} BigCode

1. Install the [nvidia-bigcode-eval](https://pypi.org/project/nvidia-bigcode-eval/) package:

```bash
pip install nvidia-bigcode-eval
```

2. Run the evaluation:

```{literalinclude} _snippets/bigcode.py
:language: python
:start-after: "## Run the evaluation"
```

:::

:::{tab-item} simple-evals

1. Install the [nvidia-simple-evals](https://pypi.org/project/nvidia-simple-evals/) package:

```bash
pip install nvidia-simple-evals
```

In the example below, we use the `AIME_2025` task, which follows the llm-as-a-judge approach for checking the output correctness.
By default, [Llama 3.3 70B](https://build.nvidia.com/meta/llama-3_3-70b-instruct) NVIDIA NIM is used for judging.

2. To run evaluation, set your [build.nvidia.com](https://build.nvidia.com/) API key as the `JUDGE_API_KEY` variable:

```bash
export JUDGE_API_KEY=...
```
To customize the judge setting, see the instructions for [NVIDIA Eval Factory package](https://pypi.org/project/nvidia-simple-evals/). 


3. Run the evaluation:

```{literalinclude} _snippets/simple_evals.py
:language: python
:start-after: "## Run the evaluation"
```

:::

:::{tab-item} safety-harness

1. Install the [nvidia-safety-harness](https://pypi.org/project/nvidia-safety-harness/) package:

```bash
pip install nvidia-safety-harness
```

2. Deploy the judge model.

In the example below, we use the `aegis_v2` task, which requires the [Llama 3.1 NemoGuard 8B ContentSafety](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html) model to assess your model's responses.

The model is available through NVIDIA NIM.
See the [instructions](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html) on deploying the judge model.

If you set up a gated judge endpoint, you must export your API key as the `JUDGE_API_KEY` variable:

```bash
export JUDGE_API_KEY=...
```
3. To access the evaluation dataset, you must authenticate with the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/quick-start#authentication).

4. Run the evaluation:

```{literalinclude} _snippets/safety.py
:language: python
:start-after: "## Run the evaluation"
```

Make sure to modify the judge configuration in the provided snippet to match your Llama 3.1 NemoGuard 8B ContentSafety endoint.

:::
::::
