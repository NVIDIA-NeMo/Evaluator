# Evaluate TensorRT-LLM checkpoints with NeMo Framework

This guide provides step-by-step instructions for evaluating TensorRT-LLM (TRTLLM) checkpoints or models. Whether the goal is to evaluate a NeMo Framework checkpoint exported to TRTLLM to ensure the correct evaluation accuracy post conversion or to evaluate a TRTLLM checkpoint directly, this guide is applicable to both the use cases.

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

Export a NeMo 2.0 checkpoint to TRTLLM using the following code snippet:

```python
from nemo_export.tensorrt_llm import TensorRTLLM

trt_llm_exporter = TensorRTLLM(model_dir="/workspace/checkpoints/llama3_8b_trtllm/")
trt_llm_exporter.export(
    nemo_checkpoint_path="/workspace/llama3_8b",
    model_type="llama",
    tensor_parallelism_size=1,
)
```

:::{note}
`deploy_ray_trtllm.py` also supports passing the NeMo checkpoint to the script which gets exported to TRTLLM for deployment with Ray. In this case, use the argument `--nemo_checkpoint_path` instead of `--trt_llm_path`. This can be followed if the TRTLLM checkpoint is not already available and you would like to export a NeMo 2.0 checkpoint to TRTLLM and evaluate it without following the separate export step mentioned above.
:::

## Evaluate TRTLLM Checkpoints

This section outlines the steps to evaluate TRTLLM checkpoints using Python commands. This method is quick and easy, making it ideal for interactive evaluations. 

Once deployment is successful, you can run evaluations using the same evaluation API described in other sections.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests.

```{literalinclude} _snippets/mmlu.py
:language: python
:start-after: "## Run the evaluation"
```