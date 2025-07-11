# Run Evaluation Using a Task Without a Pre-Defined Config

## Introduction

NVIDIA Eval Factory packages provide a unified interface and a set of pre-defined task configurations for launching evaluations.
However, you can choose to evaluate your model on a task that was not included in this set.
To do so, you must specify your tasks as `"<harness name>.<task name>"`, where the task name originates from the underlying evaluation harness. For example, NVIDIA LM-Eval is a wrapper around the LM-Evaluation-Harness.
In case of `nvidia-lm-eval`, it is a wrapper for [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness/).

Please note that when launching a custom tasks the default settings might not be optimal and you must manually provide the recommended configuration (e.g. few-shot settings).
Also, you need to determine which endpoint type should be used with the task.
Please refer to ["Evaluate NeMo 2.0 Checkpoints"](evaluation-doc.md#introduction) for more information on different endpoint types.


## Evaluate a NeMo Checkpoint with lambada_openai

In this example, we will use the `lambada_openai` task from `nvidia-lm-eval`.
The `nvidia-lm-eval` package comes pre-installed with the NeMo Framework Docker image.
If you are using a different environment, install the evaluation package:
```bash
pip install nvidia-lm-eval==25.6
```

1. Deploy your model:
```{literalinclude} ../scripts/snippets/deploy.py
:language: python
:start-after: "## Deploy"
:linenos:
```

```bash
python deploy.py
```

2. Configure and run the evaluation:

Make sure to open a new terminal within the same container to execute it.

```{literalinclude} ../scripts/snippets/lambada.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Please note that `lambada_openai` uses log-probabilities for evaluation.
To learn more about this approach, please see ["Evaluate LLMs Using Log-Probabilities"](logprobs.md).

This example uses only 10 samples.
To evaluate the full dataset, remove the `"limit_samples"` parameter.
