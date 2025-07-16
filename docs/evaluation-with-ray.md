# Multi-instance evaluations with Ray Serve

This guide explains how to deploy and evaluate NeMo FW models (trained via megatron-core backend) using Ray Serve for multi-instance evaluations.

## Introduction

Deployment with Ray Serve provides support for multiple replicas of your model across available GPUs, enabling higher throughput and better resource utilization during evaluation. This approach is particularly beneficial for evaluation scenarios where you need to process large datasets efficiently and would like to accelerate evaluation.

> **Note:** Multi-instance evaluation with Ray is currently supported only on single-node with model parallelism. Support for multi-node will be added in upcoming releases.

### Key Benefits of Ray Deployment

- **Multiple Model Replicas**: Deploy multiple instances of your model to handle concurrent requests.
- **Automatic Load Balancing**: Ray automatically distributes requests across available replicas.
- **Scalable Architecture**: Easily scale up or down based on your hardware resources.
- **Resource Optimization**: Better utilization of available GPUs.

## Deployment Configuration for Ray

To deploy your model using Ray, use the `deploy` function with `serving_backend="ray"`:

```python
from nemo_eval.api import deploy

if __name__ == "__main__":
    deploy(
        nemo_checkpoint='/workspace/llama3_8b_nemo2',
        serving_backend="ray",
        num_gpus=4,                    # Total GPUs available
        num_replicas=2,                # Number of model replicas
        tensor_parallelism_size=2,     # Tensor parallelism per replica
        pipeline_parallelism_size=1,   # Pipeline parallelism per replica
        context_parallel_size=1,       # Context parallelism per replica
        server_port=8080,              # Ray server port
    )
```

> **Note:** Adjust `num_replicas` based on the number of instances/replicas needed. Ensure that total `num_gpus` is equal to the `num_replicas` times model parallelism configuration (i.e `tensor_parallelism_size * pipeline_parallelism_size * context_parallel_size`).


## Run Evaluations on Ray-Deployed Models

Once your model is deployed with Ray, you can run evaluations using the same evaluation API as with PyTriton deployment:

```python
from nemo_eval.api import evaluate
from nemo_eval.utils.api import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions"
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation parameters
eval_params = ConfigParams(
    top_p=1,
    temperature=1,
    limit_samples=100,
    parallelism=4
)
eval_config = EvaluationConfig(type='mmlu', params=eval_params)

# Run evaluation
if __name__ == "__main__":
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```
> **Note:** To evaluate the chat endpoint, update the url by replacing `/v1/completions/` with `/v1/chat/completions/`. Additionally, set the `type` field to chat in both `ApiEndpoint` and `EvaluationConfig` to indicate a chat benchmark. A list of available chat benchmarks can be found in the  ["Evaluate checkpoints trained by NeMo Framework"](evaluation-doc.md#introduction) page.

> **Tip:** To get a performance boost from multiple replicas in Ray, increase the parallelism value in your `EvaluationConfig`. You won't see any speed improvement if  `parallelism=1`. Try setting it to a higher value, such as 4 or 8.