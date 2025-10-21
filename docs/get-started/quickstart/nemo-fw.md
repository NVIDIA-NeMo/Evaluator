---
orphan: true
---

(gs-quickstart-nemo-fw)=
# Evaluate checkpoints trained by NeMo Framework

The NeMo Framework is NVIDIA’s GPU-accelerated, end-to-end training platform for large language models (LLMs), multimodal models, and speech models. It enables seamless scaling of both pretraining and post-training workloads, from a single GPU to clusters with thousands of nodes, supporting Hugging Face/PyTorch and Megatron models. NeMo includes a suite of libraries and curated training recipes to help users build models from start to finish.

The NeMo Evaluator is integrated within NeMo Framework, offering streamlined deployment and advanced evaluation capabilities for models trained using NeMo, leveraging state-of-the-art evaluation harnesses.


## Prerequisites

- Docker with GPU support
- NeMo Framework docker container

## Quick Start

### 1. Start NeMo Framework Container

For optimal performance and user experience, use the latest version of the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent `$TAG` and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

### 2. Deploy a Model

```bash
# Deploy a NeMo checkpoint
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "/path/to/your/checkpoint" \
  --model_id megatron_model \
  --port 8080 \
  --host 0.0.0.0
```

### 3. Evaluate the Model

```python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(results)
```


## Key Features

- **Multi-Backend Deployment**: Supports PyTriton and multi-instance evaluations using the Ray Serve deployment backend
- **Production-Ready**: Supports high-performance inference with CUDA graphs and flash decoding
- **Multi-GPU and Multi-Node Support**: Enables distributed inference across multiple GPUs and compute nodes
- **OpenAI-Compatible API**: Provides RESTful endpoints aligned with OpenAI API specifications
- **Comprehensive Evaluation**: Includes state-of-the-art evaluation harnesses for academic benchmarks, reasoning benchmarks, code generation, and safety testing
- **Adapter System**: Benefits from NeMo Evaluator's Adapter System for customizable request and response processing

## Advanced Usage Patterns

### Evaluate LLMs Using Log-Probabilities

```{literalinclude} ../_snippets/arc_challenge.py
:language: python
:start-after: "## Run the evaluation"
```

### Multi-Instance Deployment with Ray

Deploy multiple instances of your model:

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

Run evaluations with increased parallelism:

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
eval_params = ConfigParams(top_p=0, temperature=0, parallelism=2)
eval_config = EvaluationConfig(type='mmlu', params=eval_params, output_dir="results")

if __name__ == "__main__":
    check_endpoint(
            endpoint_url=eval_target.api_endpoint.url,
            endpoint_type=eval_target.api_endpoint.type,
            model_name=eval_target.api_endpoint.model_id,
        )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```
