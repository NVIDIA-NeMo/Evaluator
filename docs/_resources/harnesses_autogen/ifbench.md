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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/ifbench:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:802dd90d4380a8a1e8d8723d04f3c299fbce3a0fbdf6c58f4b558db2d937e187
```

**Task Type:** `ifbench`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}}  && {% endif %} ifbench --model-url {{target.api_endpoint.url}} --model-name {{target.api_endpoint.model_id}}  --results-dir {{config.output_dir}} --inference-params max_tokens={{config.params.max_new_tokens}},temperature={{config.params.temperature}},top_p={{config.params.top_p}} --parallelism {{config.params.parallelism}} --retries {{config.params.max_retries}} {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}} {% endif %}
```

**Defaults:**
```yaml
framework_name: ifbench
pkg_name: ifbench
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 8
    task: ifbench
    temperature: 0.01
    top_p: 0.95
    extra: {}
  supported_endpoint_types:
  - chat
  type: ifbench
target:
  api_endpoint:
    stream: false
```

</details>

