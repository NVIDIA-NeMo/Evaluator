# mtbench

This page contains all evaluation tasks for the **mtbench** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [mtbench](#mtbench-mtbench)
  - Standard MT-Bench
* - [mtbench-cor1](#mtbench-mtbench-cor1)
  - Corrected MT-Bench
```

(mtbench-mtbench)=
## mtbench

Standard MT-Bench

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `mtbench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/mtbench:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:defd17e9d1b7003b9a32e38580bdd91588aede9a886e3e3d21cd539d13765fdc
```

**Task Type:** `mtbench`

**Command:**
```bash
mtbench-evaluator {% if target.api_endpoint.model_id is not none %} --model {{target.api_endpoint.model_id}}{% endif %} {% if target.api_endpoint.url is not none %} --url {{target.api_endpoint.url}}{% endif %} {% if target.api_endpoint.api_key is not none %} --api_key {{target.api_endpoint.api_key}}{% endif %} {% if config.params.request_timeout is not none %} --timeout {{config.params.request_timeout}}{% endif %} {% if config.params.max_retries is not none %} --max_retries {{config.params.max_retries}}{% endif %} {% if config.params.parallelism is not none %} --parallelism {{config.params.parallelism}}{% endif %} {% if config.params.max_new_tokens is not none %} --max_tokens {{config.params.max_new_tokens}}{% endif %} --workdir {{config.output_dir}} {% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %} {% if config.params.top_p is not none %} --top_p {{config.params.top_p}}{% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.limit_samples is not none %}--first_n {{config.params.limit_samples}}{% endif %} --generate --judge {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %}     
```

**Defaults:**
```yaml
framework_name: mtbench
pkg_name: mtbench
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: mtbench
    request_timeout: 30
    extra:
      judge:
        url: null
        model_id: gpt-4
        api_key: null
        request_timeout: 60
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 2048
  supported_endpoint_types:
  - chat
  type: mtbench
target:
  api_endpoint: {}
```

</details>


(mtbench-mtbench-cor1)=
## mtbench-cor1

Corrected MT-Bench

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `mtbench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/mtbench:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:defd17e9d1b7003b9a32e38580bdd91588aede9a886e3e3d21cd539d13765fdc
```

**Task Type:** `mtbench-cor1`

**Command:**
```bash
mtbench-evaluator {% if target.api_endpoint.model_id is not none %} --model {{target.api_endpoint.model_id}}{% endif %} {% if target.api_endpoint.url is not none %} --url {{target.api_endpoint.url}}{% endif %} {% if target.api_endpoint.api_key is not none %} --api_key {{target.api_endpoint.api_key}}{% endif %} {% if config.params.request_timeout is not none %} --timeout {{config.params.request_timeout}}{% endif %} {% if config.params.max_retries is not none %} --max_retries {{config.params.max_retries}}{% endif %} {% if config.params.parallelism is not none %} --parallelism {{config.params.parallelism}}{% endif %} {% if config.params.max_new_tokens is not none %} --max_tokens {{config.params.max_new_tokens}}{% endif %} --workdir {{config.output_dir}} {% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %} {% if config.params.top_p is not none %} --top_p {{config.params.top_p}}{% endif %} {% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %} {% if config.params.limit_samples is not none %}--first_n {{config.params.limit_samples}}{% endif %} --generate --judge {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %}     
```

**Defaults:**
```yaml
framework_name: mtbench
pkg_name: mtbench
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: mtbench-cor1
    request_timeout: 30
    extra:
      judge:
        url: null
        model_id: gpt-4
        api_key: null
        request_timeout: 60
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 2048
      args: --judge_reference_model gpt-4-0125-preview
  supported_endpoint_types:
  - chat
  type: mtbench-cor1
target:
  api_endpoint: {}
```

</details>

