# Adding on-demand evaluation packages

## Introduction
The NeMo Framework docker image comes with `nvidia-lm-eval` pre-installed.
However, you can add more evaluation frameworks by installing additional NVIDIA Eval Factory packages.

For every package you need to follow these steps:

1. Install the required package

2. Deploy your model:

```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:linenos:
```

Run the deployment in the background:
```bash
python deploy.py &
```

3. (Optional) Export the required environment variables. 

4. Run the evalution of your choice.

Below you can find examples for enabling and launching evaluations for different frameworks.

## Enable BFCL

First, install the [nvidia-bfcl](https://pypi.org/project/nvidia-bfcl/) package:
```bash
pip install nvidia-bfcl==25.6
```

Then you can run the evaluation:
```{literalinclude} ../scripts/snippets/bfcl.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable garak

First, install the [nvidia-eval-factory-garak](https://pypi.org/project/nvidia-eval-factory-garak/) package:
```bash
pip install nvidia-eval-factory-garak==25.6
```

Then you can run the evaluation:
```{literalinclude} ../scripts/snippets/garak.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable BigCode

First, install the [nvidia-bigcode](https://pypi.org/project/nvidia-bigcode/) package:
```bash
pip install nvidia-bigcode==25.6
```

Then you can run the evaluation:
```{literalinclude} ../scripts/snippets/bigcode.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable simple-evals

First, install the [nvidia-simple-evals](https://pypi.org/project/nvidia-simple-evals/) package:
```bash
pip install nvidia-simple-evals==25.6
```

In the example below we use the `AIME_2025` task, which follows the llm-as-a-judge approach for checking the output correctness.
By default, [Llama 3.3 70B](https://build.nvidia.com/meta/llama-3_3-70b-instruct) NVIDIA NIM is used for judging.
In order to run evaluation, set your [build.nvidia.com](https://build.nvidia.com/) API key as the `JUDGE_API_KEY` variable:

```bash
export JUDGE_API_KEY=...
```
For customizing the judge setting see the instructions for [NVIDIA Eval Factory package](https://pypi.org/project/nvidia-simple-evals/). 


Then you can run the evaluation:
```{literalinclude} ../scripts/snippets/simple_evals.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

## Enable safety-harness

First, install the [nvidia-safety-harness](https://pypi.org/project/nvidia-safety-harness/) package:
```bash
pip install nvidia-safety-harness==25.6
```

In the example below we use the `aegis_v2` task, which requiress [Llama 3.1 NemoGuard 8B ContentSafety](https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html) model for assesing the your model's responses.

The model is available as NVIDIA NIM and you can either deploy it (see the [Instructions]((https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/getting-started.html))) or try the preview version at [build.nvidia.com](https://build.nvidia.com/).

If you use a gated endpoint (e.g. the preview NIM described above), please make sure to export your API key as the `JUDGE_API_KEY` variable:

```bash
export JUDGE_API_KEY=...
```
Also you must authenticate to the [Hugging Face Hub](https://huggingface.co/docs/huggingface_hub/quick-start#authentication) to access the dataset used for evaluation.

Then you can run the evaluation:
```{literalinclude} ../scripts/snippets/safety.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Make sure to modify the judge configuration in the provided snippet to match the Llama 3.1 NemoGuard 8B ContentSafety endoint of your choice:

```python
    params={
        "extra": {
            "judge": {
                "model_id": "my-llama-3.1-nemoguard-8b-content-safety-endpoint",
                "url": "http://my-hostname:8000/v1/chat/completions",
            }
        }
    }
```