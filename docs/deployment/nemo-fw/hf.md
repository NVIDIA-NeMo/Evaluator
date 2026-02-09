# Evaluate Automodel Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating checkpoints trained using the NeMo Framework with the Automodel backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

Here, we focus on benchmarks within the `lm-evaluation-harness` that depend on text generation. Evaluation on log-probability-based benchmarks is available in [Evaluate Automodel Checkpoints on Log-probability benchmarks](#evaluate-automodel-checkpoints-on-log-probability-benchmarks).

## Deploy Automodel Checkpoints

This section outlines the steps to deploy Automodel checkpoints using Python commands.

Automodel checkpoint deployment uses Ray Serve as the serving backend. It also offers an OpenAI API (OAI)-compatible endpoint, similar to deployments of checkpoints trained with the Megatron Core backend. An example deployment command is shown below.

```{literalinclude} _snippets/deploy_hf.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

The `--model_path` can refer to either a local checkpoint path or a Hugging Face model ID, as shown in the example above. In the example above, checkpoint deployment uses the `vLLM` backend. To enable accelerated inference, install `vLLM` in your environment. To install `vLLM` inside the NeMo Framework container, follow the steps below as shared in [Export-Deploy's README](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=readme-ov-file#install-tensorrt-llm-vllm-or-trt-onnx-backend:~:text=cd%20/opt/export%2ddeploy%0auv%20sync%20%2d%2dinexact%20%2d%2dlink%2dmode%20symlink%20%2d%2dlocked%20%2d%2dextra%20vllm%20%24(cat%20/opt/uv_args.txt)):

```shell
cd /opt/Export-Deploy
uv sync --inexact --link-mode symlink --locked --extra vllm $(cat /opt/uv_args.txt)
```

To install `vLLM` outside of the NeMo Framework container, follow the steps mentioned [here](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=readme-ov-file#install-tensorrt-llm-vllm-or-trt-onnx-backend:~:text=install%20transformerengine%20%2b%20vllm).

:::{note}
25.11 release of NeMo Framework container comes with `vLLM` pre-installed and its not necessary to explicitly install it. However for all previous releases, please refer to the instructions above to install `vLLM` inside the NeMo Framework container.
:::

If you prefer to evaluate the Automodel checkpoint without using the `vLLM` backend, remove the `--use_vllm_backend` flag from the command above.

:::{note}
To speed up evaluation using multiple instances, increase the `num_replicas` parameter.
For additional guidance, refer to {ref}`nemo-fw-ray`.
:::

## Evaluate Automodel Checkpoints

This section outlines the steps to evaluate Automodel checkpoints using Python commands. This method is quick and easy, making it ideal for interactive evaluations. 

Once deployment is successful, you can run evaluations using the {ref}`lib-core` API.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests.

```{literalinclude} _snippets/mmlu.py
:language: python
:start-after: "## Run the evaluation"
```

## Evaluate Automodel Checkpoints on Log-probability Benchmarks

To evaluate Automodel checkpoints on benchmarks that require log-probabilities, use the same deployment command provided in [Deploy Automodel Checkpoints](#deploy-automodel-checkpoints). These benchmarks are supported by both the `vLLM` backend (enabled via the `--use_vllm_backend` flag) and by directly deploying the Automodel checkpoint.

For evaluation, you must specify the path to the `tokenizer` and set the `tokenizer_backend` parameter as shown below. The `tokenizer` files are located within the checkpoint directory.

```{literalinclude} _snippets/arc_challenge_hf.py
:language: python
:start-after: "## Run the evaluation"
```

## Evaluate Automodel Checkpoints on Chat Benchmarks

To evaluate Automodel checkpoints on chat benchmarks you need the chat endpoint (`/v1/chat/completions/`). The deployment command provided in [Deploy Automodel Checkpoints](#deploy-automodel-checkpoints) also exposes the chat endpoint, and the same command can be used for evaluating on chat benchmarks.

For evaluation, update the URL by replacing `/v1/completions/` with `/v1/chat/completions/` as shown below. Additionally, set the `type` field to `"chat"` to indicate a chat benchmark.

```{literalinclude} _snippets/ifeval.py
:language: python
:start-after: "## Run the evaluation"
```
