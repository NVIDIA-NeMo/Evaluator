(run-eval-reasoning)=
# Evaluation of Reasoning Models

Reasoning models require a distinct approach compared to standard language models. Their outputs are typically longer, may contain dedicated reasoning tokens, and are more susceptible to generating loops or repetitive sequences. Evaluating these models effectively requires custom parameter settings and careful handling of generation constraints.

## Before You Start

Ensure you have:

- **Model Endpoint**: An OpenAI-compatible API endpoint for your model (completions or chat). See {ref}`deployment-testing-compatibility` for snippets you can use to test your endpoint.
- **API Access**: Valid API key if your endpoint requires authentication
- **Installed Packages**: NeMo Evaluator or access to evaluation containers


## Recommended Settings

### Generation Settings

Below are recommended generation settings for some popular reasoning-optimized models. These configurations should be included in the **model card**:

| Model               | Temperature | Top-p  | Top-k  | 
|---------------------|-------------|--------|--------|
| **NVIDIA Nemotron** | 0.6         | 0.95   | —      |
| **DeepSeek R1**     | 0.6         | 0.95   | —      |
| **Qwen 230B**       | 0.6         | 0.95   | 20     |
| **Phi-4 Reasoning** | 0.8         | 0.95   | 50     |


### Token Configuration

- `max_new_tokens` must be **significantly increased** for reasoning tasks as it includes the length of both reasoning trace and the final answer.
- Check the model card to see settings recommended by the model creators.
- It is important to observe if the specified `max_new_tokens` is enough for the model to finish reasoning.

:::{tip}
You can verify successful reasoning completion in the logs via the {ref}`interceptor-reasoning` Interceptor, for example:

```
[I 2025-12-02T16:14:28.257] Reasoning tracking information reasoning_words=1905 original_content_words=85 updated_content_words=85 reasoning_finished=True reasoning_started=True reasoning_tokens=unknown updated_content_tokens=unknown logger=ResponseReasoningInterceptor request_id=ccff76b2-2b85-4eed-a9d0-2363b533ae58
```
:::

## Reasoning Output Formats

Reasoning models produce outputs that contain both the **reasoning trace** (the model's step-by-step thinking process) and the **final answer**. The reasoning trace typically includes intermediate thoughts, calculations, and logical steps before arriving at the conclusion.

There are two main ways to structure reasoning output:

### 1. Wrapped with reasoning tokens

e.g.

```
... </think>
```

```
<think> ... </think>
```

or

```
<reason> ... </reason><final></final>
```

Most of the benchmarks expect only the final answer to be present in model's response.
If your model endpoint replies with reasoning trace present in the main content, it needs to be removed from the assistant messages.
You can do it using the {ref}`interceptor-reasoning` Interceptor.
The interceptor will remove reasoning trace from the content and (optionally) track statistics for reasoning traces.

:::{note}
The `ResponseReasoningInterceptor` is by default configured for the `...</think>` and `<think> ...</think>` format. If your model uses these special tokens, you do not need to modify anything in your configuration.
:::

### 2. Returned as `reasoning_content` field in messages output

If your model is deployed with e.g. vLLM, sglang or NIM, the reasoning part of the model's output is likely returned in the separate `reasoning_content` field in messages output (see [vLLM documentation](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html) and [sglang documentation](https://sgl-project.github.io/advanced_features/separate_reasoning.html)).

In the messages returned by the endpoint, there are:

- `reasoning_content`: The reasoning part of the output.
- `content`: The content of the final answer.

Conversely to the first method, this setup does not require any extra response parsing.
However, in some benchmarks, errors may appear if the reasoning has not finished and the benchmark does not support empty answers in `content`.

#### Enabling reasoning parser in vLLM

To enable the `reasoning_content` field in vLLM, you need to pass the `--reasoning-parser` argument to the vLLM server.
In NeMo Evaluator Launcher, you can do this via `deployment.extra_args`:

```yaml
deployment:
  hf_model_handle: Qwen/Qwen3-Next-80B-A3B-Thinking
  extra_args: "--reasoning-parser deepseek_r1"
```

Available reasoning parsers depend on your vLLM version. Common options include `deepseek_r1` for models using `<think>...</think>` format.
See the [vLLM reasoning outputs documentation](https://docs.vllm.ai/en/stable/features/reasoning_outputs.html) for details.

---

## Control the Reasoning Effort

Some models allow turning reasoning on/off or setting its level of effort. There are usually 2 ways of doing it:

- **Special instruction in the system prompt**
- **Extra parameters passed to the chat_template**

:::{tip}
Check the model card and documentation of the deployment of your choice to see how you can control the reasoning effort for your model.
If there are several options available, it is recommended to use the dedicated chat template parameters over the system prompt.
:::

### Control reasoning with the system prompt

In this example we will use the [NVIDIA-Nemotron-Nano-9B-v2](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2/modelcard) model.
This model allows you to control the reasoning effort by including `/think` or `/no_think` in the system prompt, e.g.:


```json
{
  "model": "nvidia/nvidia-nemotron-nano-9b-v2",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. /think"},
    {"role": "user", "content": "What is 2+2?"}
  ],
  "temperature": 0.6,   
  "top_p": 0.95,
  "max_tokens": 32768
}
```

When launching the evaluation, we can use the {ref}`interceptor-system-messages` Interceptor to add `/think` or `/no_think` to the system prompt.


```yaml
config:
  params:
    temperature: 0.6
    top_p: 0.95
    max_new_tokens: 32768  # for reasoning + final answer
target:
  api_endpoint:
    adapter_config:
      process_reasoning_traces: true # strips reasoning tokens and collects reasoning stats
      use_system_prompt: true # turn reasoning on with special system prompt
      custom_system_prompt: >-
        "/think"
```


### Control reasoning with additional parameters

In this example we will use the [Granite-3.3-8B-Instruct](https://build.nvidia.com/ibm/granite-3_3-8b-instruct/modelcard) model.
Conversely to NVIDIA-Nemotron-Nano-9B-v2, this model allows you to turn the reasoning on with an additional `thinking` parameter passed to the chat template:

```json
{
  "model": "ibm/granite-3.3-8b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2?"
    }
  ],
  "temperature": 0.2,
  "top_p": 0.7,
  "max_tokens": 8192,
  "seed": 42,
  "stream": true,
  "chat_template_kwargs": {
    "thinking": true
  }
}
```

When running the evaluation, use the {ref}`interceptor-payload-modification` Interceptor to add this parameter to benchmarks' requests:

```yaml
config:
  params:
    temperature: 0.6
    top_p: 0.95
    max_new_tokens: 32768  # for reasoning + final answer
target:
  api_endpoint:
    adapter_config:
      process_reasoning_traces: true 
      params_to_add:
        chat_template_kwargs:
          thinking: true
```


## Benchmarks for Reasoning

Reasoning models excel at tasks that require multi-step thinking, logical deduction, and complex problem-solving. The following benchmark categories are particularly well-suited for evaluating reasoning capabilities:


- **CoT tasks**: e.g., AIME, Math, GPQA-diamond
- **Coding**: e.g., scicodebench, livedocebench


:::{tip}
When evaluating your model on a task that does not require step-by-step thinking, consider turning the reasoning off or lowering the thinking budget.
:::


## Full Working Example

### Run the evaluation

An example config is available in `packages/nemo-evaluator-launcher/examples/local_reasoning.yaml`:

```{literalinclude} ../../../packages/nemo-evaluator-launcher/examples/local_reasoning.yaml
:language: yaml
:start-after: "[docs-start-snippet]"
```

To launch the evaluation, run:

```bash
export NGC_API_KEY=nvapi-...
nemo-evaluator-launcher run \
  --config packages/nemo-evaluator-launcher/examples/local_reasoning.yaml
```

### Analyze the artifacts

NeMo Evaluator produces several artifacts for analysis after evaluation completion.
The primary output file is `results.yaml`, which stores the metrics produced by the benchmark (see {ref}`evaluation-output` for more details).

The `eval_factory_metrics.json` file provides valuable insights into your model's behavior.
When the reasoning interceptor is enabled, this file contains a `reasoning` key that stores statistics about reasoning traces in your model's responses:

```json
"reasoning": {
    "description": "Reasoning statistics saved during processing",
    "total_responses": 3672,
    "responses_with_reasoning": 2860,
    "reasoning_finished_count": 2860,
    "reasoning_finished_ratio": 1.0,
    "reasoning_started_count": 2860,
    "reasoning_unfinished_count": 0,
    "avg_reasoning_words": 153.21,
    "avg_original_content_words": 192.17,
    "avg_updated_content_words": 38.52,
    "max_reasoning_words": 806,
    "max_original_content_words": 863,
    "max_updated_content_words": 863,
    "max_reasoning_tokens": null,
    "avg_reasoning_tokens": null,
    "max_updated_content_tokens": null,
    "avg_updated_content_tokens": null,
    "total_reasoning_words": 561696,
    "total_original_content_words": 705555,
    "total_updated_content_words": 140999,
    "total_reasoning_tokens": 0,
    "total_updated_content_tokens": 0
  },
```

In the example above, the model used reasoning for 2860 out of 3672 responses (approximately 78%).

The matching values for `reasoning_started_count` and `reasoning_finished_count` (and `reasoning_unfinished_count` being 0) indicate that the `max_new_tokens` parameter was set sufficiently high, allowing the model to complete all reasoning traces without truncation.

These statistics also enable cost analysis for reasoning operations.
While the endpoint in this example does not return reasoning token usage statistics (the `*_tokens` fields are null or zero), you can still analyze computational cost using the word count metrics from the responses.

For more information on available artifacts, see {ref}`evaluation-output`.
