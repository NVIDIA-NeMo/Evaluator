# tooltalk

This page contains all evaluation tasks for the **tooltalk** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [tooltalk](#tooltalk-tooltalk)
  - tooltalk
```

(tooltalk-tooltalk)=
## tooltalk

tooltalk

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `tooltalk`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/tooltalk:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:d87254b0784060facdbe107f9a8b9768fe5f857e0e6575b6f536b6e5ccf5e48a
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}API_KEY=${{target.api_endpoint.api_key}}{% endif %} python -m tooltalk.evaluation.evaluate_{{'openai' if 'azure' in target.api_endpoint.url or 'api.openai' in target.api_endpoint.url else 'nim'}} --dataset data/easy --database data/databases --model {{target.api_endpoint.model_id}} {% if config.params.max_new_tokens is not none %}--max_new_tokens {{config.params.max_new_tokens}}{% endif %} {% if config.params.temperature is not none %}--temperature {{config.params.temperature}}{% endif %} {% if config.params.top_p is not none %}--top_p {{config.params.top_p}}{% endif %} --api_mode all --output_dir {{config.output_dir}} --url {{target.api_endpoint.url}} {% if config.params.limit_samples is not none %}--first_n {{config.params.limit_samples}}{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    task: tooltalk
  type: tooltalk
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

