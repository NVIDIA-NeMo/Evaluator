(how-to-reasoning)=
# Remove Reasoning Traces

This guide walks you through configuring NeMo Evaluator Launcher for evaluating reasoning models. It shows how to:

- adjust sampling parameters
- remove reasoning traces from the answer
- controlling thinking budget

ensuring accurate benchmark evaluation.

:::{tip}
Need more in-depth explanation? See the {ref}`run-eval-reasoning` guide.
:::

## Before You Start

Ensure you have:

- **Model Endpoint**: An OpenAI-compatible API reasoning endpoint for your model (completions or chat). See {ref}`deployment-testing-compatibility` for snippets you can use to test your endpoint and {ref}`run-eval-reasoning` for details on reasoning models.
- **API Access**: Valid API key if your endpoint requires authentication
- **Installed Packages**: NeMo Evaluator or access to evaluation containers

## Prepare your config file

### Configure the Evaluation

1. Select tasks:

```yaml
evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
    - name: mgsm
```

2. Adjust sampling parameters for a reasoning model, e.g.:

```yaml
evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
    - name: mgsm
  nemo_evaluator_config:
    config:
      params:
        temperature: 0.6
        top_p: 0.95
        max_new_tokens: 32768  # for reasoning + final answer
        request_timeout: 3600  # long timeout to account for thinking time
        parallelism: 1  # single parallel request to avoid overloading the server
```

3. Enable Reasoning Interceptor to remove reasoning traces from the model's responses:

```yaml
evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
    - name: mgsm
  nemo_evaluator_config:
    config:
      params:
        temperature: 0.6
        top_p: 0.95
        max_new_tokens: 32768  # for reasoning + final answer
        request_timeout: 3600  # long timeout to account for thinking time
        parallelism: 1  # single parallel request to avoid overloading the server
    target:
      api_endpoint:
        adapter_config:
          interceptors:
            - name: endpoint
            - name: reasoning
```

In this example we will use [NVIDIA-Nemotron-Nano-9B-v2](https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2), which produces reasoning trace in a `<think>...</think>` format.
If your model uses a different formatting, make sure to configure the interceptor as shown in {ref}`run-eval-reasoning`.

4. (Optional) Modify the request to turn the reasoning on.

In this example we work with an endpoint that requires "/think" to be present in the system message to enable reasoning.
We will use the Interceptor to add it to the request.


Adjust the example below to match your endpoint (see detailed instructions in {ref}`run-eval-reasoning`).


```yaml
evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
    - name: mgsm
  nemo_evaluator_config:
    target:
      api_endpoint:
        adapter_config:
          interceptors:
            - name: system_message
              config:
                system_message: "/think"
            - name: endpoint
            - name: reasoning
```

### Select your execution backend and deployment specification

For the purpose of this example, we will use local execution without deployment.
See other How-to guides to adjust this example to your needs.

1. Configure local executor

```yaml
defaults:
  - execution: local
  - _self_

execution:
  output_dir: nel-results
```

2. Configure target endpoint

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: nel-results

target:
  api_endpoint:
    # see https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2 for endpoint details
    model_id: nvidia/nvidia-nemotron-nano-9b-v2
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY # API Key with access to build.nvidia.com
```

### The Full Config

Combine all components into a config file for your experiment:

```yaml
defaults:
  - execution: local
  - deployment: none
  - _self_

execution:
  output_dir: nel-results

target:
  api_endpoint:
    # see https://build.nvidia.com/nvidia/nvidia-nemotron-nano-9b-v2 for endpoint details
    model_id: nvidia/nvidia-nemotron-nano-9b-v2
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY # API Key with access to build.nvidia.com

evaluation:
  tasks:
    - name: simple_evals.mmlu_pro
    - name: mgsm
  nemo_evaluator_config:
    config:
      params:
        temperature: 0.6
        top_p: 0.95
        max_new_tokens: 32768  # for reasoning + final answer
        request_timeout: 3600  # long timeout to account for thinking time
        parallelism: 1  # single parallel request to avoid overloading the server
    target:
      api_endpoint:
        adapter_config:
          interceptors:
            - name: system_message
              config:
                system_message: "/think"
            - name: endpoint
            - name: reasoning
```

## Verify and execute your experiment

1. Save the prepared config in a file, e.g. `nemotron_eval.yaml`


2. (Recommended) Inspect the configuration with `--dry_run`


```bash
export NGC_API_KEY=nvapi-your-key
nemo-evaluator-launcher run --config nemotron_eval.yaml --dry_run
```

3. (Recommended) Run a short experiment with 10 samples per benchmark to verify your config

```bash
export NGC_API_KEY=nvapi-your-key
nemo-evaluator-launcher run --config nemotron_eval.yaml \
  -o +evaluation.nemo_evaluator_config.config.params.limit_samples=10
```

:::{tip}
If everything works correctly you should see logs from the `ResponseReasoningInterceptor` similar to the ones below:

```bash
[I 2025-12-02T16:14:28.257] Reasoning tracking information reasoning_words=1905 original_content_words=85 updated_content_words=85 reasoning_finished=True reasoning_started=True reasoning_tokens=unknown updated_content_tokens=unknown logger=ResponseReasoningInterceptor request_id=ccff76b2-2b85-4eed-a9d0-2363b533ae58
```

:::

4. Run the full experiment

```bash
export NGC_API_KEY=nvapi-your-key
nemo-evaluator-launcher run --config nemotron_eval.yaml
```

5. Analyze the metrics and reasoning statistics

After evaluation completes, check these key artifacts:

- **`results.yaml`**: Contains the benchmark metrics (see {ref}`evaluation-output`)
- **`eval_factory_metrics.json`**: Contains reasoning statistics under the `reasoning` key, including:
  - `responses_with_reasoning`: How many responses included reasoning traces
  - `reasoning_finished_count` vs `reasoning_started_count`: If these match, your `max_new_tokens` was sufficient
  - `avg_reasoning_words` and other word- and tokens count metrics: Use these for cost analysis

:::{tip}
For detailed explanation of reasoning statistics and artifacts, see {ref}`run-eval-reasoning`.
:::

