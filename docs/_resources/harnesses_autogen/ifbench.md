# ifbench

This page contains all evaluation tasks for the **ifbench** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [ifbench](#ifbench-ifbench)
  - ifbench
```

(ifbench-ifbench)=
## ifbench

ifbench

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `ifbench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/ifbench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:38bb1ab96c495591b3cebbcf9bb87251a1a1a46abe36b2e067a8217d973fe6c5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}}  && {% endif %} ifbench --model-url {{target.api_endpoint.url}} --model-name {{target.api_endpoint.model_id}}  --results-dir {{config.output_dir}} --inference-params max_tokens={{config.params.max_new_tokens}},temperature={{config.params.temperature}},top_p={{config.params.top_p}} --parallelism {{config.params.parallelism}} --retries {{config.params.max_retries}} {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0.01
    top_p: 0.95
    parallelism: 8
    max_retries: 5
    task: ifbench
  type: ifbench
target:
  api_endpoint:
    stream: false

```

</details>

