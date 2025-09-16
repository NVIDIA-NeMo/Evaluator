# Evaluate Automodel Checkpoints Trained by NeMo Framework

This guide provides step-by-step instructions for evaluating checkpoints trained using the NeMo Framework with the Automodel backend. This section specifically covers evaluation with [nvidia-lm-eval](https://pypi.org/project/nvidia-lm-eval/), a wrapper around the [
lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main) tool.

Here, we focus on benchmarks within the `lm-evaluation-harness` that depend on text generation. For a detailed comparison between generation-based and log-probability-based benchmarks, refer to ["Evaluate Checkpoints Trained by NeMo Framework"](evaluation-doc.md).

> **Note:** The current support for evaluation of Automodel checkpoints is limited to generation benchmarks. Support for  log-probability benchmarks will be added in upcoming releases.

## Deploy Automodel Checkpoints

This section outlines the steps to deploy Automodel checkpoints using Python commands.

Automodel checkpoint deployment uses Ray Serve as the serving backend. It also offers an OpenAI API (OAI)-compatible endpoint, similar to deployments of checkpoints trained with the Megatron Core backend. An example deployment command is shown below.

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

The `hf_model_id_path` can refer to either a local checkpoint path or a Hugging Face model ID, as shown in the example above. By default, Automodel checkpoint deployment uses the `vLLM` backend. To enable accelerated inference, install `vLLM` in your environment—either within the NeMo Framework container or externally—using the command `pip install vllm`. If you prefer to evaluate the Automodel checkpoint without using the `vLLM` backend, set `use_vllm_backend` to `False` in the deploy method.

> **Note:** Ensure that `ray` is specified as the `serving_backend`, as Automodel checkpoint evaluation is supported only with the Ray backend. To speed up evaluation using multiple instances, increase the `num_replicas` parameter.
For additional guidance, refer to ["Use Ray Serve for Multi-Instance Evaluations"](evaluation-with-ray.md).
## Evaluate Automodel Checkpoints

This section outlines the steps to evaluate Automodel checkpoints using Python commands. This method is quick and easy, making it ideal for interactive evaluations.
Once deployment is successful, you can run evaluations using the same evaluation API described in other sections, such as the ["Evaluate Models Locally on Your Workstation"](evaluation-doc.md#evaluate-models-locally-on-your-workstation) section.

Before starting the evaluation, it’s recommended to use the [`check_endpoint`](https://github.com/NVIDIA-NeMo/Eval/blob/main/src/nemo_eval/utils/base.py) function to verify that the endpoint is responsive and ready to accept requests.

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
