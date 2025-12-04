# garak

This page contains all evaluation tasks for the **garak** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [garak](#garak-garak)
  - garak
```

(garak-garak)=
## garak

garak

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `garak`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/garak:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:eaf083b45b1114b22577b2f9606c5717d05bd2e31d50bb5fdea6b350ddb6e2c7
```

**Task Type:** `garak`

**Command:**
```bash
cat > garak_config.yaml << 'EOF'
{% if config.params.extra.seed is not none %}run:
  seed: {{config.params.extra.seed}}{% endif %}
plugins:
  {% if config.params.extra.probes is not none %}probe_spec: {{config.params.extra.probes}}{% endif %}
  extended_detectors: true
  model_type: {% if target.api_endpoint.type == "completions" %}nim.NVOpenAICompletion{% elif target.api_endpoint.type == "chat" %}nim.NVOpenAIChat{% endif %}
  model_name: {{target.api_endpoint.model_id}}
  generators:
    nim:
      uri: {{target.api_endpoint.url | replace('/chat/completions', '') | replace('/completions', '')}}
      {% if config.params.temperature is not none %}temperature: {{config.params.temperature}}{% endif %}
      {% if config.params.top_p is not none %}top_p: {{config.params.top_p}}{% endif %}
      {% if config.params.max_new_tokens is not none %}max_tokens: {{config.params.max_new_tokens}}{% endif %}
system:
  parallel_attempts: {{config.params.parallelism}}
  lite: false
EOF
{% if target.api_endpoint.api_key is not none %}
export NIM_API_KEY=${{target.api_endpoint.api_key}} &&
{% else %}
export NIM_API_KEY=dummy &&
{% endif %}
export XDG_DATA_HOME={{config.output_dir}} &&
garak --config garak_config.yaml --report_prefix=results

```

**Defaults:**
```yaml
framework_name: garak
pkg_name: garak
config:
  params:
    max_new_tokens: 150
    parallelism: 32
    task: garak
    temperature: 0.1
    top_p: 0.7
    extra:
      probes: null
      seed: null
  supported_endpoint_types:
  - chat
  - completions
  type: garak
target:
  api_endpoint: {}
```

</details>

