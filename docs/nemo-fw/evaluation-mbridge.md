# Evaluate Megatron Bridge Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/index.html) checkpoints trained using the NeMo Framework with the Megatron Core backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

First, we focus on benchmarks within the `lm-evaluation-harness` that depend on text generation. For a detailed comparison between generation-based and log-probability-based benchmarks, refer to ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md). Evaluation on log-probability-based benchmarks is available in the subsequent section [Evaluate Megatron Bridge Checkpoints on Log-probability benchmarks](#evaluate-megatron-bridge-checkpoints-on-log-probability-benchmarks).

## Deploy Megatron Bridge Checkpoints

To evaluate a checkpoint saved during pretraining or fine-tuning with [Megatron-Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/recipe-usage.html), provide the path to the saved checkpoint using the `--megatron_checkpoint` flag in the deployment command below. Otherwise, Hugging Face checkpoints can be converted to Megatron Bridge using the single shell command:

```bash
huggingface-cli login --token <your token>
python -c "from megatron.bridge import AutoBridge; AutoBridge.import_ckpt('meta-llama/Llama-3-8B','/workspace/mbridge_llama3_8b/')"
```

The deployment scripts are available inside the [`/opt/Export-Deploy/scripts/deploy/nlp/`](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts/deploy/nlp) directory. Below is an example command for deployment. It uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to Megatron Bridge format using the command shared above.

```shell
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint "/workspace/mbridge_llama3_8b/iter_0000000" \
  --model_id "megatron_model" \
  --port 8080 \                          # Ray server port
  --num_gpus 4 \                         # Total GPUs available
  --num_replicas 2 \                     # Number of model replicas
  --tensor_model_parallel_size 2 \       # Tensor parallelism per replica
  --pipeline_model_parallel_size 1 \     # Pipeline parallelism per replica
  --context_parallel_size 1              # Context parallelism per replica
```

> **Note:** Megatron Bridge creates checkpoints in directories named `iter_N`, where *N* is the iteration number. Each `iter_N` directory contains model weights and related artifacts. When using a checkpoint, make sure to provide the path to the appropriate `iter_N` directory. Hugging Face checkpoints converted for Megatron Bridge are typically stored in a directory named `iter_0000000`, as shown in the command above.

> **Note:** Megatron Bridge deployment for evaluation is supported only with Ray Serve and not PyTriton.

## Evaluate Megatron Bridge Checkpoints

Once deployment is successful, you can run evaluations using the same evaluation API described in other sections, such as the ["Evaluate Models Locally on Your Workstation"](evaluation-doc.md#evaluate-models-locally-on-your-workstation) section.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests.

```python
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(top_p=0, temperature=0, limit_samples=2, parallelism=1)
eval_config = EvaluationConfig(type='mmlu', params=eval_params, output_dir="results")

if __name__ == "__main__":
    check_endpoint(
            endpoint_url=eval_target.api_endpoint.url,
            endpoint_type=eval_target.api_endpoint.type,
            model_name=eval_target.api_endpoint.model_id,
        )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```

## Evaluate Megatron Bridge Checkpoints on Log-probability Benchmarks

To evaluate Megatron Bridge checkpoints on benchmarks that require log-probabilities, use the same deployment command provided in [Deploy Megatron Bridge Checkpoints](#deploy-megatron-bridge-checkpoints).

For evaluation, you must specify the path to the `tokenizer` and set the `tokenizer_backend` parameter as shown below. The `tokenizer` files are located within the `tokenizer` directory of the checkpoint.

```python
from nemo_evaluator.api import check_endpoint, evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(top_p=0, temperature=0, limit_samples=1, parallelism=1,
                            extra={
                                "tokenizer": '/workspace/mbridge_llama3_8b/iter_0000000/tokenizer',
                                "tokenizer_backend": "huggingface",
                                },
                            )
eval_config = EvaluationConfig(type='arc_challenge', params=eval_params, output_dir="results")

if __name__ == "__main__":
    check_endpoint(
            endpoint_url=eval_target.api_endpoint.url,
            endpoint_type=eval_target.api_endpoint.type,
            model_name=eval_target.api_endpoint.model_id,
        )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```