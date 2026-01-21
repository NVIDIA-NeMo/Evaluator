# Evaluate Megatron Bridge Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) checkpoints trained using the NeMo Framework with the Megatron Core backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

First, we focus on benchmarks within the `lm-evaluation-harness` that depend on text generation. Evaluation on log-probability-based benchmarks is available in the subsequent section [Evaluate Megatron Bridge Checkpoints on Log-probability benchmarks](#evaluate-megatron-bridge-checkpoints-on-log-probability-benchmarks).

## Deploy Megatron Bridge Checkpoints

To evaluate a checkpoint saved during pretraining or fine-tuning with [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html), provide the path to the saved checkpoint using the `--megatron_checkpoint` flag in the deployment command below. Otherwise, Hugging Face checkpoints can be converted to Megatron Bridge using the single shell command:

```bash
huggingface-cli login --token <your token>
python -c "from megatron.bridge import AutoBridge; AutoBridge.import_ckpt('meta-llama/Meta-Llama-3-8B','/workspace/mbridge_llama3_8b/')"
```

The deployment scripts are available inside the [`/opt/Export-Deploy/scripts/deploy/nlp/`](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts/deploy/nlp) directory. Below is an example command for deployment. It uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to Megatron Bridge format using the command shared above.

```{literalinclude} _snippets/deploy_mbridge.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

:::{note}
Megatron Bridge creates checkpoints in directories named `iter_N`, where *N* is the iteration number. Each `iter_N` directory contains model weights and related artifacts. When using a checkpoint, make sure to provide the path to the appropriate `iter_N` directory. Hugging Face checkpoints converted for Megatron Bridge are typically stored in a directory named `iter_0000000`, as shown in the command above.
:::

:::{note}
Megatron Bridge deployment for evaluation is supported only with Ray Serve and not PyTriton.
:::

## Evaluate Megatron Bridge Checkpoints

Once deployment is successful, you can run evaluations using the NeMo Evaluator API. See {ref}`lib-core` for more details.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests.

```{literalinclude} _snippets/mmlu.py
:language: python
:start-after: "## Run the evaluation"
```

## Evaluate Megatron Bridge Checkpoints on Log-probability Benchmarks

To evaluate Megatron Bridge checkpoints on benchmarks that require log-probabilities, use the same deployment command provided in [Deploy Megatron Bridge Checkpoints](#deploy-megatron-bridge-checkpoints).

For evaluation, you must specify the path to the `tokenizer` and set the `tokenizer_backend` parameter as shown below. The `tokenizer` files are located within the `tokenizer` directory of the checkpoint.

```{literalinclude} _snippets/arc_challenge_mbridge.py
:language: python
:start-after: "## Run the evaluation"
```