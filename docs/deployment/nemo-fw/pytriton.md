# Use PyTriton Server for Evaluations

This guide explains how to deploy and evaluate NeMo Framework models, trained with the Megatron-Core backend, using PyTriton to serve the model.

## Introduction

Deploymement with PyTriton serving backend provides high-performance inference through the NVIDIA Triton Inference Server, with OpenAI API compatibility via a FastAPI interface.
It supports model parallelism across single-node and multi-node configurations, facilitating deployment of large models that cannot fit into a single device.



## Key Benefits of PyTriton Deployment

- **Multi-Node support**: Deploy large models on multiple nodes using pipeline-, tensor-, context- or expert-parallelism.
- **Automatic Requests Batching**: PyTriton automatically groups your requests into batches for efficient inference.



## Deploy Models Using PyTriton

The deployment scripts are available inside [`/opt/Export-Deploy/scripts/deploy/nlp/`](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts/deploy/nlp) directory.
The example command below uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to NeMo format. To evaluate a checkpoint saved during [pretraining](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#pretraining) or [fine-tuning](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html#fine-tuning), provide the path to the saved checkpoint using the `--nemo_checkpoint` flag in the command below.

```shell
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --nemo_checkpoint "/workspace/llama3_8b_nemo2" \
  --triton_model_name "megatron_model" \
  --server_port 8080 \
  --num_gpus 1 \
  --max_batch_size 4 \
  --inference_max_seq_length 4096
```

When working with a larger model, you can use model parallelism to distribute the model across available devices.
In the example below we deploy the Qwen3-235B-A22B model in the NeMo format with 4 nodes (each with 8 devices):

```bash
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py \
  --nemo_checkpoint "/workspace/qwen3-235b-a22b_nemo2" \
  --triton_model_name "megatron_model" \
  --server_port 8080 \
  --num_nodes 4 \
  --num_gpus 8 \
  --tensor_model_parallel_size 4  \
  --pipeline_model_parallel_size 8 \
  --context_parallel_size 1 \
  --expert_model_parallel_size 4 \
  --account_for_embedding_in_pipeline_split \
  --account_for_loss_in_pipeline_split \
  --max_batch_size 4 \
  --inference_max_seq_length 4096
```

Make sure to adjust the parameters to match your available resource and model architecture.


## Run Evaluations on PyTriton-Deployed Models

<!-- TODO: replace with links to API reference -->
The entry point for evaluation is the [`evaluate`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/evaluate.py) function. To run evaluations on the deployed model, use the following command. Make sure to open a new terminal within the same container to execute it. For longer evaluations, it is advisable to run both the deploy and evaluate commands in tmux sessions to prevent the processes from being terminated unexpectedly and aborting the runs.
It is recommended to use [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests before starting the evaluation.

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

To evaluate the chat endpoint, update the url by replacing `/v1/completions/` with `/v1/chat/completions/`. Additionally, set the `type` field to `"chat"` in both `ApiEndpoint` and `EvaluationConfig` to indicate a chat benchmark.

To evaluate log-probability benchmarks (e.g., `arc_challenge`), run the following code snippet after deployment.
<!-- For a comparison between generation benchmarks and log-probability benchmarks, refer to the ["Evaluate Checkpoints Trained by NeMo Framework"](index.md) section. -->

Make sure to open a new terminal within the same container to execute it.


```{literalinclude} ../../scripts/snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Note that in the example above, you must provide a path to the tokenizer:

```
        "extra": {
            "tokenizer": "/workspace/llama3_8b_nemo2/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
        },
```


<!-- TODO: replace with links to API reference -->
Please refer to [`deploy_inframework_triton.py`](https://github.com/NVIDIA-NeMo/Export-Deploy/blob/main/scripts/deploy/nlp/deploy_inframework_triton.py) script and [`evaluate`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/evaluate.py) function to review all available argument options, as the provided commands are only examples and do not include all arguments or their default values. For more detailed information on the arguments used in the ApiEndpoint and ConfigParams classes for evaluation, see [`api_dataclasses`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/api/api_dataclasses.py) submodule.


:::{tip}
If you encounter a TimeoutError on the eval client side, please increase the `request_timeout` parameter in `ConfigParams` class to a larger value like `1000` or `1200` seconds (the default is 300).
:::

