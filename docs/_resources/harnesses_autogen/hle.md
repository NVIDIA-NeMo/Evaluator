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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/hle:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:b1d4456951b751baeb7b8354e8d58f32be4e0f88f5dbeaabac34cb5c1832c78c
```

**Command:**
```bash
hle_eval --dataset=cais/hle --model_name={{target.api_endpoint.model_id}} --model_url={{target.api_endpoint.url}}  --temperature={{config.params.temperature}} --top_p={{config.params.top_p}} --timeout={{config.params.request_timeout}}  {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --output_dir={{config.output_dir}}  {% if target.api_endpoint.api_key is not none %}--api_key_name={{target.api_endpoint.api_key}}{% endif %} --max_retries={{config.params.max_retries}} --num_workers={{config.params.parallelism}}  --max_new_tokens={{config.params.max_new_tokens}} --text_only --generate --judge
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    temperature: 0.0
    top_p: 1.0
    request_timeout: 600.0
    max_new_tokens: 4096
    max_retries: 30
    parallelism: 100
    task: hle
  type: hle
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

