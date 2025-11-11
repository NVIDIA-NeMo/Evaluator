# safety-eval

This page contains all evaluation tasks for the **safety-eval** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [aegis_v2](#safety-eval-aegis-v2)
  - Aegis V2 without evaluating reasoning traces. This version is used by the NeMo Safety Toolkit.
* - [aegis_v2_reasoning](#safety-eval-aegis-v2-reasoning)
  - Aegis V2 with evaluating reasoning traces.
* - [wildguard](#safety-eval-wildguard)
  - Wildguard
```

(safety-eval-aegis-v2)=
## aegis_v2

Aegis V2 without evaluating reasoning traces. This version is used by the NeMo Safety Toolkit.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `safety-eval`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/safety-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:c53b77274383e1b784a704b31db04e40618e55d40b57c5d2a3faf6bbdfacf509
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}}  && {% endif %} {% if config.params.extra.judge.api_key is not none %}export JUDGE_API_KEY=${{config.params.extra.judge.api_key}} && {% endif %} safety-eval  --model-name  {{target.api_endpoint.model_id}} --model-url {{target.api_endpoint.url}} --model-type {{target.api_endpoint.type}}  --judge-url  {{config.params.extra.judge.url}}   --results-dir {{config.output_dir}}   --eval {{config.params.task}}  --mut-inference-params max_tokens={{config.params.max_new_tokens}},temperature={{config.params.temperature}},top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},concurrency={{config.params.parallelism}},retries={{config.params.max_retries}} --judge-inference-params concurrency={{config.params.extra.judge.parallelism}},retries={{config.params.max_retries}}  {% if config.params.extra.dataset is defined and config.params.extra.dataset %} --dataset {{config.params.extra.dataset}}{% endif %} {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}} {% endif %} {% if config.params.extra.judge.model_id is not none %} --judge-model-name {{config.params.extra.judge.model_id}} {% endif %} {% if config.type == "aegis_v2_reasoning" %} {% if config.params.extra.evaluate_reasoning_traces  %} --evaluate-reasoning-traces {% endif %} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  - completions
  params:
    limit_samples: null
    max_new_tokens: 6144
    temperature: 0.6
    top_p: 0.95
    parallelism: 8
    max_retries: 5
    request_timeout: 30
    extra:
      judge:
        url: null
        model_id: null
        api_key: null
        parallelism: 32
        request_timeout: 60
        max_retries: 16
      evaluate_reasoning_traces: false
    task: aegis_v2
  type: aegis_v2
target:
  api_endpoint:
    stream: false

```

</details>


(safety-eval-aegis-v2-reasoning)=
## aegis_v2_reasoning

Aegis V2 with evaluating reasoning traces.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `safety-eval`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/safety-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:c53b77274383e1b784a704b31db04e40618e55d40b57c5d2a3faf6bbdfacf509
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}}  && {% endif %} {% if config.params.extra.judge.api_key is not none %}export JUDGE_API_KEY=${{config.params.extra.judge.api_key}} && {% endif %} safety-eval  --model-name  {{target.api_endpoint.model_id}} --model-url {{target.api_endpoint.url}} --model-type {{target.api_endpoint.type}}  --judge-url  {{config.params.extra.judge.url}}   --results-dir {{config.output_dir}}   --eval {{config.params.task}}  --mut-inference-params max_tokens={{config.params.max_new_tokens}},temperature={{config.params.temperature}},top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},concurrency={{config.params.parallelism}},retries={{config.params.max_retries}} --judge-inference-params concurrency={{config.params.extra.judge.parallelism}},retries={{config.params.max_retries}}  {% if config.params.extra.dataset is defined and config.params.extra.dataset %} --dataset {{config.params.extra.dataset}}{% endif %} {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}} {% endif %} {% if config.params.extra.judge.model_id is not none %} --judge-model-name {{config.params.extra.judge.model_id}} {% endif %} {% if config.type == "aegis_v2_reasoning" %} {% if config.params.extra.evaluate_reasoning_traces  %} --evaluate-reasoning-traces {% endif %} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  - completions
  params:
    limit_samples: null
    max_new_tokens: 6144
    temperature: 0.6
    top_p: 0.95
    parallelism: 8
    max_retries: 5
    request_timeout: 30
    extra:
      judge:
        url: null
        model_id: null
        api_key: null
        parallelism: 32
        request_timeout: 60
        max_retries: 16
      evaluate_reasoning_traces: true
    task: aegis_v2
  type: aegis_v2_reasoning
target:
  api_endpoint:
    stream: false

```

</details>


(safety-eval-wildguard)=
## wildguard

Wildguard

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `safety-eval`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/safety-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:c53b77274383e1b784a704b31db04e40618e55d40b57c5d2a3faf6bbdfacf509
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}}  && {% endif %} {% if config.params.extra.judge.api_key is not none %}export JUDGE_API_KEY=${{config.params.extra.judge.api_key}} && {% endif %} safety-eval  --model-name  {{target.api_endpoint.model_id}} --model-url {{target.api_endpoint.url}} --model-type {{target.api_endpoint.type}}  --judge-url  {{config.params.extra.judge.url}}   --results-dir {{config.output_dir}}   --eval {{config.params.task}}  --mut-inference-params max_tokens={{config.params.max_new_tokens}},temperature={{config.params.temperature}},top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},concurrency={{config.params.parallelism}},retries={{config.params.max_retries}} --judge-inference-params concurrency={{config.params.extra.judge.parallelism}},retries={{config.params.max_retries}}  {% if config.params.extra.dataset is defined and config.params.extra.dataset %} --dataset {{config.params.extra.dataset}}{% endif %} {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}} {% endif %} {% if config.params.extra.judge.model_id is not none %} --judge-model-name {{config.params.extra.judge.model_id}} {% endif %} {% if config.type == "aegis_v2_reasoning" %} {% if config.params.extra.evaluate_reasoning_traces  %} --evaluate-reasoning-traces {% endif %} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  - completions
  params:
    limit_samples: null
    max_new_tokens: 6144
    temperature: 0.6
    top_p: 0.95
    parallelism: 8
    max_retries: 5
    request_timeout: 30
    extra:
      judge:
        url: null
        model_id: null
        api_key: null
        parallelism: 32
        request_timeout: 60
        max_retries: 16
    task: wildguard
  type: wildguard
target:
  api_endpoint:
    stream: false

```

</details>

