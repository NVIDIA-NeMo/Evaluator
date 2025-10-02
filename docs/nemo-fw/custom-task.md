# Run Evaluation Using a Task Without a Pre-Defined Config

## Introduction

NVIDIA Eval Factory packages provide a unified interface and a set of pre-defined task configurations for launching evaluations.
However, you can choose to evaluate your model on a task that was not included in this set.
To do so, you must specify your tasks as `"<harness name>.<task name>"`, where the task name originates from the underlying evaluation harness. For example, NVIDIA LM-Eval is a wrapper around the LM-Evaluation-Harness.
In case of `nvidia-lm-eval`, it is a wrapper for [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/).

Please note that when launching custom tasks, the default settings may not be optimal. You must manually provide the recommended configuration (e.g., few-shot settings). Additionally, you need to determine which endpoint type is appropriate for the task. 
Please refer to ["Deploy and Evaluate NeMo Checkpoints"](evaluation-doc.md#deploy-and-evaluate-nemo-checkpoints) for more information on different endpoint types.


## Evaluate a NeMo Checkpoint with lambada_openai

In this example, we will use the `lambada_openai` task from `nvidia-lm-eval`.
The `nvidia-lm-eval` package comes pre-installed with the NeMo Framework Docker image.
If you are using a different environment, install the evaluation package:

```bash
pip install nvidia-lm-eval
```

1. Deploy your model:

```{literalinclude} ../../scripts/snippets/deploy.sh
:language: shell
:start-after: "## Deploy"
```

You can verify if the server is ready for accepting requests with the following function:
```python
from nemo_evaluator.api import check_endpoint

check_endpoint(
    endpoint_url="http://0.0.0.0:8080/v1/completions/",
    endpoint_type="completions",
    model_name="megatron_model",
)
```


2. Configure and run the evaluation:

Be sure to launch a new terminal inside the same container before running the command.

```{literalinclude} ../../scripts/snippets/lambada.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Please note that `lambada_openai` uses log-probabilities for evaluation.
To learn more about this approach, please see ["Evaluate LLMs Using Log-Probabilities"](logprobs.md).

This example uses only 10 samples.
To evaluate the full dataset, remove the `"limit_samples"` parameter.
