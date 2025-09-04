# Evaluate Automodel Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating checkpoints trained using the NeMo Framework with the Automodel backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

Here, we focus on benchmarks in the `lm-evaluation-harness` that rely on text generation.
For more details on generation benchmarks v/s logprob benchmarks, refer to ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md).

> **Note:** The current support for evaluation of automodel checkpoints is limited to generation benchmarks and support for logprob benchmarks will be added in upcoming releases.

### Deploy and Evaluate Automodel checkpoints

This section outlines the steps to deploy and evaluate automodel checkpoint using Python commands. This method is quick and easy, making it ideal for interactive evaluations. 

Deployment of Automodel checkpoints uses Ray serve as the serving backend. Deployment of Automodel checkpoint also provides OpenAI API (OAI) compatible endpoint similar to deployment of checkpoints trained with Megatron Core backend. Below is an example command for deployment.

```python
from nemo_eval.api import deploy

if __name__ == "__main__":
    deploy(
        hf_model_id_path='meta-llama/Llama-3.1-8B',
        serving_backend='ray',
        max_input_len=4096,
        max_batch_size=4,
        server_port=8080,
        num_gpus=1,
        num_replicas=1)
```

`hf_model_id_path` can be a path to the local checkpoint or HuggingFace model id as shown in the example above.
Deployment of Automodel checkpoints uses `vllm` backend by default. Please install `vllm` in your environment (NeMo FW container or outside) with the command `pip install vllm` to benefit from accelerated inference with `vlm` backend. If you would like to evaluate Automodel checkpoint directly without using the `vllm` backend, please set `use_vllm_backend` to `False` in the deploy method.

> **Note:** Make sure to provide `ray` as the `serving_backend` as the Automodel checkpoint evaluation is supported only with `ray` backend. Also increase `num_replicas` for faster evaluation with multiple instances. More details on this in ["Use Ray Serve for Multi-Instance Evaluations"](evaluation-with-ray.md)

Once deployment is successful, you can run evaluations using the same evaluation API as described in other sections like ["Evaluate Models Locally on Your Workstation"](evaluation-doc.md#evaluate-models-locally-on-your-workstation) section. 
It is recommended to use [`check_endpoint`](https://github.com/NVIDIA-NeMo/Eval/blob/main/src/nemo_eval/utils/base.py) function to verify that the endpoint is responsive and ready to accept requests before starting the evaluation.

```python
from nemo_eval.utils.base import check_endpoint
from nvidia_eval_commons.core.evaluate import evaluate
from nvidia_eval_commons.api.api_dataclasses import EvaluationConfig, ApiEndpoint, EvaluationTarget, ConfigParams

# Configure the evaluation target
api_endpoint = ApiEndpoint(
    url="http://0.0.0.0:8080/v1/completions/",
    type="completions",
    model_id="megatron_model",
)
eval_target = EvaluationTarget(api_endpoint=api_endpoint)
eval_params = ConfigParams(top_p=0, temperature=0, limit_samples=2, parallelism=1)
eval_config = EvaluationConfig(type='mmlu', params=eval_params)

if __name__ == "__main__":
    check_endpoint(
            endpoint_url=eval_target.api_endpoint.url,
            endpoint_type=eval_target.api_endpoint.type,
            model_name=eval_target.api_endpoint.model_id,
        )
    evaluate(target_cfg=eval_target, eval_cfg=eval_config)
```