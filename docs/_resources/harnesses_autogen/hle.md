# hle

This page contains all evaluation tasks for the **hle** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [hle](#hle-hle)
  - hle
```

(hle-hle)=
## hle

hle

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `hle`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/hle:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:991af6d4348d61b2d707250a3da6c8aa61f93fc54726a30ad8905cf99f455d21
```

**Task Type:** `hle`

**Command:**
```bash
hle_eval --dataset=cais/hle --model_name={{target.api_endpoint.model_id}} --model_url={{target.api_endpoint.url}}  --temperature={{config.params.temperature}} --top_p={{config.params.top_p}} --timeout={{config.params.request_timeout}}  {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --output_dir={{config.output_dir}}  {% if target.api_endpoint.api_key is not none %}--api_key_name={{target.api_endpoint.api_key}}{% endif %} --max_retries={{config.params.max_retries}} --num_workers={{config.params.parallelism}}  --max_new_tokens={{config.params.max_new_tokens}} --text_only --generate --judge
```

**Defaults:**
```yaml
framework_name: hle
pkg_name: hle
config:
  params:
    max_new_tokens: 4096
    max_retries: 30
    parallelism: 100
    task: hle
    temperature: 0.0
    request_timeout: 600
    top_p: 1.0
    extra: {}
  supported_endpoint_types:
  - chat
  type: hle
target:
  api_endpoint: {}
```

</details>

