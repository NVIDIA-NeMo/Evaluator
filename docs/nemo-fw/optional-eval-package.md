# Add On-Demand Evaluation Packages

This guide explains how to extend the NeMo evaluation environment by adding optional NVIDIA Eval Factory packages. It walks through installation, setup, and execution steps for various packages such as BFCL, garak, BigCode, simple-evals, and safety-harness, each enabling specialized model assessments.

The following diagram illustrates the architecture of the Eval ecosystem, showing how different evaluation harnesses integrate with NeMo Eval:

```
┌──────────────────────┐
│                      │
│lm-evaluation-harness ◄─────────┐
│                      │         │      ┌─────────────────────┐
└──────────────────────┘         │      │                     │
                                 │      │ NVIDIA Eval Factory │
┌──────────────────────┐         │      │ =================== │
│                      ◄─────────┼──────┼                     ◄───┐      ┌────────────────────┐
│     simple-evals     │    packaging   │ unified interface   │ model    │                    │
│                      │    of eval     │ for LLM evaluation  │ querying │     NeMo Eval      │
└──────────────────────┘    harnesses   │                     │   └──────┼ ================== │
                                 │      └─────────────────────┘          │                    │
  .                              │                                       │   LLM evaluation   │
  .                   ◄──────────┼                                ┌──────┼ with server-client │
  .                              │      ┌────────────────────┐  model    │      approach      │
                                 │      │                    │  serving  │                    │
                                 │      │ NeMo Export-Deploy │    │      └────────────────────┘
┌──────────────────────┐         │      │ ================== │    │
|                      │         │      │                    │    │
│        garak         ◄─────────┘      │  model deployment  ◄────┘
│                      │                │  for NeMo and HF   │
└──────────────────────┘                │                    │
                                        └────────────────────┘
```

## Add New Evaluation Frameworks
The NeMo Framework Docker image comes with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/) pre-installed.
However, you can add more evaluation methods by installing additional NVIDIA Eval Factory packages.

For each package, follow these steps:

1. Install the required package.

2. Deploy your model:

```{literalinclude} ../scripts/snippets/deploy.sh
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
Note that all examples use only a subset of samples.
To run the evaluation on the whole dataset, remove the `"limit_samples"` parameter.

## Enable BFCL

1. Install the [nvidia-bfcl](https://pypi.org/project/nvidia-bfcl/) package:

```bash
pip install nvidia-bfcl
```

2. Run the evaluation:

```{literalinclude} ../scripts/snippets/bfcl.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable garak

1. Install the [nvidia-eval-factory-garak](https://pypi.org/project/nvidia-eval-factory-garak/) package:

```bash
pip install nvidia-eval-factory-garak
```

2. Run the evaluation:

```{literalinclude} ../scripts/snippets/garak.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable BigCode

1. Install the [nvidia-bigcode-eval](https://pypi.org/project/nvidia-bigcode-eval/) package:

```bash
pip install nvidia-bigcode-eval
```

2. Run the evaluation:

```{literalinclude} ../scripts/snippets/bigcode.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable simple-evals

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

```{literalinclude} ../scripts/snippets/simple_evals.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable safety-harness

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

```{literalinclude} ../scripts/snippets/safety.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Make sure to modify the judge configuration in the provided snippet to match your Llama 3.1 NemoGuard 8B ContentSafety endoint:

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