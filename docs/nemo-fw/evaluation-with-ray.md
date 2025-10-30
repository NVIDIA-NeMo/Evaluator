# Use Ray Serve for Multi-Instance Evaluations

This guide explains how to deploy and evaluate Megatron Bridge models, trained in NeMo Framework with the Megatron-Core backend, using Ray Serve to enable multi-instance evaluation across available GPUs.

## Introduction

Deployment with Ray Serve provides support for multiple replicas of your model across available GPUs, enabling higher throughput and better resource utilization during evaluation. This approach is particularly beneficial for evaluation scenarios where you need to process large datasets efficiently and would like to accelerate evaluation.

> **Note:** Multi-instance evaluation with Ray is currently supported only on a single node with model parallelism. Support for multi-node will be added in upcoming releases.

## Key Benefits of Ray Deployment

- **Multiple Model Replicas**: Deploy multiple instances of your model to handle concurrent requests.
- **Automatic Load Balancing**: Ray automatically distributes requests across available replicas.
- **Scalable Architecture**: Easily scale up or down based on your hardware resources.
- **Resource Optimization**: Better utilization of available GPUs.

## Deploy Models Using Ray Serve

To deploy your model using Ray, use the `deploy_ray_inframework.py` script available inside the [`/opt/Export-Deploy/scripts/deploy/nlp/`](https://github.com/NVIDIA-NeMo/Export-Deploy/tree/main/scripts/deploy/nlp) directory. Below is an example command for deployment. It uses a Hugging Face LLaMA 3 8B checkpoint that has been converted to Megatron Bridge format.

```shell
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --megatron_checkpoint "/workspace/llama3_8b/iter_0000000" \
  --model_id "megatron_model" \
  --port 8080 \                          # Ray server port
  --num_gpus 4 \                         # Total GPUs available
  --num_replicas 2 \                     # Number of model replicas
  --tensor_model_parallel_size 2 \       # Tensor parallelism per replica
  --pipeline_model_parallel_size 1 \     # Pipeline parallelism per replica
  --context_parallel_size 1              # Context parallelism per replica
```

> **Note:** Adjust `num_replicas` based on the number of instances/replicas needed. Ensure that total `num_gpus` is equal to the `num_replicas` times model parallelism configuration (i.e., `tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size`).

> **Note:** To evaluate NeMo 2.0 checkpoints, replace the `--megatron_checkpoint` flag in the deployment command above with `--nemo_checkpoint` and provide the path for the NeMo checkpoint.


## Run Evaluations on Ray-Deployed Models

Once your model is deployed with Ray, you can run evaluations using the same evaluation API as with PyTriton deployment. It is recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests before starting the evaluation.

To evaluate on generation benchmarks, use the code snippet below:

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
> **Note:** To evaluate the chat endpoint, update the url by replacing `/v1/completions/` with `/v1/chat/completions/`. Additionally, set the `type` field to `"chat"` in both `ApiEndpoint` and `EvaluationConfig` to indicate a chat benchmark. A list of available chat benchmarks can be found in the ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md#evaluate-checkpoints-trained-by-nemo-framework) page.

To evaluate log-probability benchmarks (e.g., `arc_challenge`), run the following code snippet after deployment.
For a comparison between generation benchmarks and log-probability benchmarks, refer to the ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md) section.

Make sure to open a new terminal within the same container to execute it.


```{literalinclude} ../../scripts/snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
:linenos:
```

Note that in the example above, you must provide a path to the tokenizer:

```
        "extra": {
            "tokenizer": "/workspace/llama3_8b/iter_0000000/tokenizer",
            "tokenizer_backend": "huggingface",
        },
```

For more details on log-probability benchmarks, refer to ["Evaluate LLMs Using Log-Probabilities"](logprobs.md).

> **Tip:** To get a performance boost from multiple replicas in Ray, increase the parallelism value in your `EvaluationConfig`. You won't see any speed improvement if `parallelism=1`. Try setting it to a higher value, such as 4 or 8.
