(eval-custom-tasks)=
# Tasks Not Explicitly Defined by Framework Definition File

## Introduction

NeMo Evaluator provides a unified interface and a curated set of pre-defined task configurations for launching evaluations.
These task configurations are specified in the [Framework Definition File (FDF)](../about/concepts/framework-definition-file.md) to provide a simple and standardized way of running evaluations, with minimum user-provided input required.

However, you can choose to evaluate your model on a task that was not explicitly included in the FDF.
To do so, you must specify your task as `"<harness name>.<task name>"`, where the task name originates from the underlying evaluation harness, and ensure that all of the task parameters (e.g., sampling parameters, few-shot settings) are specified correctly.
Additionally, you need to determine which [endpoint type](../deployment/bring-your-own-endpoint/testing-endpoint-oai-compatibility.md) is appropriate for the task.

## Run Evaluation

In this example, we will use the [PolEmo 2.0](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/polemo2) task from LM Evaluation Harness.
This task consists of consumer reviews in Polish and assesses sentiment analysis abilities.
It requires a "completions" endpoint and has the sampling parameters defined as a part of [task configuration](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/polemo2/polemo2_in.yaml) of the underlying harness.

:::{note}
Make sure to review the task configuration in the underlying harness and ensure the sampling parameters are defined and match your preffered way of running the benchmark.

You can configure the evaluation using the `params` field in the `EvaluationConfig`.
:::

### 1. Prepare the Environment

Start `lm-evaluation-harness` Docker container:

```bash
docker run --rm -it nvcr.io/nvidia/eval-factory/lm-evaluation-harness:{{ docker_compose_latest }}
```

or install `nemo-evaluator` and `nvidia-lm-eval` Python package in your environment of choice:

```bash
pip install nemo-evaluator nvidia-lm-eval
```

### 2. Run the Evaluation

```{literalinclude} _snippets/polemo2.py
:language: python
:start-after: "## Run the evaluation"
```
