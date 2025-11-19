(nemo-fw-ray)=
# Use Ray Serve for Multi-Instance Evaluations

This guide explains how to deploy and evaluate NeMo Framework models, trained with the Megatron-Core backend, using Ray Serve to enable multi-instance evaluation across available GPUs.

## Introduction

Deployment with Ray Serve provides support for multiple replicas of your model across available GPUs, enabling higher throughput and better resource utilization during evaluation. This approach is particularly beneficial for evaluation scenarios where you need to process large datasets efficiently and would like to accelerate evaluation.

:::{note}
Multi-instance evaluation with Ray is currently supported only on a single node with model parallelism. Support for multi-node will be added in upcoming releases.
:::

## Key Benefits of Ray Deployment

- **Multiple Model Replicas**: Deploy multiple instances of your model to handle concurrent requests.
- **Automatic Load Balancing**: Ray automatically distributes requests across available replicas.
- **Scalable Architecture**: Easily scale up or down based on your hardware resources.
- **Resource Optimization**: Better utilization of available GPUs.

## Deploy Models Using Ray Serve

To deploy your model using Ray, use the `deploy_ray_inframework.py` script from [NeMo Export-Deploy](https://github.com/NVIDIA-NeMo/Export-Deploy):

```shell
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "meta-llama/Llama-3.1-8B" \
  --model_id "megatron_model" \
  --port 8080 \                          # Ray server port
  --num_gpus 4 \                         # Total GPUs available
  --num_replicas 2 \                     # Number of model replicas
  --tensor_model_parallel_size 2 \       # Tensor parallelism per replica
  --pipeline_model_parallel_size 1 \     # Pipeline parallelism per replica
  --context_parallel_size 1              # Context parallelism per replica
```

:::{note}
Adjust `num_replicas` based on the number of instances/replicas needed. Ensure that total `num_gpus` is equal to the `num_replicas` times model parallelism configuration (i.e., `tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size`).
:::


## Run Evaluations on Ray-Deployed Models

Once your model is deployed with Ray, you can run evaluations using the same evaluation API as with PyTriton deployment. It is recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/packages/nemo-evaluator/src/nemo_evaluator/core/utils.py) function to verify that the endpoint is responsive and ready to accept requests before starting the evaluation.

To evaluate on generation benchmarks use the code snippet below:

```{literalinclude} _snippets/mmlu.py
:language: python
:start-after: "## Run the evaluation"
```

To evaluate the chat endpoint, update the url by replacing `/v1/completions/` with `/v1/chat/completions/`. Additionally, set the `type` field to `"chat"` in both `ApiEndpoint` and `EvaluationConfig` to indicate a chat benchmark.
<!-- A list of available chat benchmarks can be found in the ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md#evaluate-checkpoints-trained-by-nemo-framework) page. -->


To evaluate log-probability benchmarks (e.g., `arc_challenge`), run the following code snippet after deployment.
<!-- For a comparison between generation benchmarks and log-probability benchmarks, refer to the ["Evaluate Checkpoints Trained by NeMo Framework"](index.md) section. -->

Make sure to open a new terminal within the same container to execute it.


```{literalinclude} _snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
```

Note that in the example above, you must provide a path to the tokenizer:

```python
        extra={
            "tokenizer": "/workspace/llama3_8b_nemo2/context/nemo_tokenizer",
            "tokenizer_backend": "huggingface",
        },
```

<!-- For more details on log-probability benchmarks, refer to ["Evaluate LLMs Using Log-Probabilities"](logprobs.md). -->

:::{tip}
To get a performance boost from multiple replicas in Ray, increase the parallelism value in your `EvaluationConfig`. You won't see any speed improvement if `parallelism=1`. Try setting it to a higher value, such as 4 or 8.
:::