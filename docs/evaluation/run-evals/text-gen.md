(text-gen)=

# Text Generation Evaluation

Text generation evaluation is the primary method for assessing LLM capabilities where models produce natural language responses to prompts. This approach evaluates the quality, accuracy, and appropriateness of generated text across various tasks and domains.


:::{tip}
In the example below we use the `gpqa_diamond` benchmark, but the instructions provided apply to all text generation tasks, such as:

- `mmlu`
- `mmlu_pro`
- `ifeval`
- `gsm8k`
- `mgsm`
- `mbpp`

:::

## Before You Start

Ensure you have:

- **Model Endpoint**: An OpenAI-compatible API endpoint for your model (completions or chat). See {ref}`deployment-testing-compatibility` for snippets you can use to test your endpoint.
- **API Access**: Valid API key if your endpoint requires authentication
- **Installed Packages**: NeMo Evaluator or access to evaluation containers

## Evaluation Approach

In text generation evaluation:

1. **Prompt Construction**: Models receive carefully crafted prompts (questions, instructions, or text to continue)
2. **Response Generation**: Models generate natural language responses using their trained parameters
3. **Response Assessment**: Generated text is evaluated for correctness, quality, or adherence to specific criteria
4. **Metric Calculation**: Numerical scores are computed based on evaluation criteria

This differs from **log-probability evaluation** where models assign confidence scores to predefined choices.
For log-probability methods, see the {ref}`logprobs`.


## Use NeMo Evaluator Launcher

Use an example config for evaluating the [Meta Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) model:

```{literalinclude} ../../../packages/nemo-evaluator-launcher/examples/local_basic.yaml
:language: yaml
:start-after: "[docs-start-snippet]"
```


To launch the evaluation, run:

```bash

export HF_TOKEN_FOR_GPQA_DIAMOND=hf_your-token-here  # GPQA is a gated dataset
export NGC_API_KEY=nvapi-your-token-here  # API Key with access to build.nvidia.com

nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/local_basic.yaml
```


## Use NeMo Evaluator

Start `simple-evals` docker container:

```bash
docker run --rm -it nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
```

or install `nemo-evaluator` and `nvidia-simple-evals` Python package in your environment of choice:

```bash
pip install nemo-evaluator nvidia-simple-evals
```

### Run with CLI

```bash
export HF_TOKEN_FOR_GPQA_DIAMOND=hf_your-token-here  # GPQA is a gated dataset
export NGC_API_KEY=nvapi-your-token-here  # API Key with access to build.nvidia.com

# Run evaluation
nemo-evaluator run_eval \
    --eval_type gpqa_diamond \
    --model_id meta/llama-3.2-3b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name NGC_API_KEY \
    --output_dir ./llama_3_1_8b_instruct_results
```

### Run with Python API

```python
# set env variables before entering Python:
# export HF_TOKEN_FOR_GPQA_DIAMOND=hf_your-token-here  # GPQA is a gated dataset
# export NGC_API_KEY=nvapi-your-token-here  # API Key with access to build.nvidia.com

from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint, EvaluationConfig, EvaluationTarget, ConfigParams, EndpointType
)

# Configure target endpoint
api_endpoint = ApiEndpoint(
    url="https://integrate.api.nvidia.com/v1/chat/completions",
    type=EndpointType.CHAT,
    model_id="meta/llama-3.2-3b-instruct",
    api_key="NGC_API_KEY"  # variable name storing the key
)
target = EvaluationTarget(api_endpoint=api_endpoint)

# Configure evaluation task
config = EvaluationConfig(
    type="gpqa_diamond",
    output_dir="./llama_3_1_8b_instruct_results"
)

# Execute evaluation
results = evaluate(target_cfg=target, eval_cfg=config)
```
