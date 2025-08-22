# Quick Start

## 1. Deploy a Model

```python
from nemo_eval.api import deploy

# Deploy a NeMo checkpoint
deploy(
    nemo_checkpoint="/path/to/your/checkpoint",
    serving_backend="pytriton",  # or "ray"
    server_port=8080,
    num_gpus=1,
    max_input_len=4096,
    max_batch_size=8
)
```

## 2. Evaluate the Model

```python
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import ApiEndpoint, EvaluationConfig, EvaluationTarget

# Configure evaluation
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    model_id="megatron_model"
)
target = EvaluationTarget(api_endpoint=api_endpoint)
config = EvaluationConfig(type="gsm8k", output_dir="results")

# Run evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
print(results)
```
