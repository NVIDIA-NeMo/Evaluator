# Evaluate TensorRT-LLM checkpoints with NeMo Framework

This guide provides step-by-step instructions for evaluating TensorRT-LLM (TRTLLM) checkpoints or models inside NeMo Framework.

This guide focuses on benchmarks within the `lm-evaluation-harness` that depend on text generation. For a detailed comparison between generation-based and log-probability-based benchmarks, refer to {ref}`eval-run`.

:::{note}
Evaluation on log-probability-based benchmarks for TRTLLM models is currently planned for a future release.
:::

## Deploy TRTLLM Checkpoints

This section outlines the steps to deploy TRTLLM checkpoints using Python commands.  

TRTLLM checkpoint deployment uses Ray Serve as the serving backend. It also offers an OpenAI API (OAI)-compatible endpoint, similar to deployments of checkpoints trained with the Megatron Core backend. An example deployment command is shown below.

```{literalinclude} _snippets/deploy_trtllm.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

## Evaluate TRTLLM Checkpoints

This section outlines the steps to evaluate TRTLLM checkpoints using Python commands. This method is quick and easy, making it ideal for interactive evaluations. 

Once deployment is successful, you can run evaluations using the same evaluation API described in other sections.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests.

```{literalinclude} _snippets/mmlu.py
:language: python
:start-after: "## Run the evaluation"
```