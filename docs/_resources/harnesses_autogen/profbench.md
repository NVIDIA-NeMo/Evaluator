# profbench

This page contains all evaluation tasks for the **profbench** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [llm_judge](#profbench-llm-judge)
  - Run LLM judge on provided ProfBench reports and score them
* - [report_generation](#profbench-report-generation)
  - Generate professional reports and evaluate them (full pipeline)
```

(profbench-llm-judge)=
## llm_judge

Run LLM judge on provided ProfBench reports and score them

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `profbench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/profbench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:251d1b47c52fbfed94837ef6ab5e0cca61ac218bb82d01b0e4314fb10264a9ef
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} {% if config.params.extra.run_generation %}
  python -m profbench.run_report_generation \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --folder {{config.output_dir}}{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.extra.version is not none %} --version {{config.params.extra.version}}{% endif %}{% if config.params.extra.upload_documents %} --upload-documents{% endif %}{% if config.params.extra.web_search %} --web-search{% endif %}{% if config.params.extra.reasoning %} --reasoning{% endif %}{% if config.params.extra.reasoning_effort is not none %} --reasoning-effort {{config.params.extra.reasoning_effort}}{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  GENERATION_OUTPUT=$(ls -t {{config.output_dir}}/*.jsonl | head -1) && 
{% endif %} {% if config.params.extra.run_judge_generated %}
  python -m profbench.run_best_llm_judge_on_generated_reports \
    --filename $GENERATION_OUTPUT \
    --api-key $API_KEY \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --output-folder {{config.output_dir}}/judgements{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  JUDGE_OUTPUT=$(ls -t {{config.output_dir}}/judgements/*.jsonl | head -1) && 
  python -m profbench.score_report_generation $JUDGE_OUTPUT
{% endif %} {% if config.params.extra.run_judge_provided %}
  python -m profbench.run_llm_judge_on_provided_reports \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --folder {{config.output_dir}}{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.extra.reasoning %} --reasoning{% endif %}{% if config.params.extra.reasoning_effort is not none %} --reasoning-effort {{config.params.extra.reasoning_effort}}{% endif %}{% if config.params.extra.debug %} --debug{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  JUDGE_OUTPUT=$(ls -t {{config.output_dir}}/*.jsonl | head -1) && 
  python -m profbench.score_llm_judge $JUDGE_OUTPUT
{% endif %}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 600
    extra:
      run_generation: false
      run_judge_generated: false
      run_judge_provided: true
      library: openai
      version: lite
      upload_documents: false
      web_search: false
      reasoning: false
      reasoning_effort: null
      debug: false
  type: llm_judge
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(profbench-report-generation)=
## report_generation

Generate professional reports and evaluate them (full pipeline)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `profbench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/profbench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:251d1b47c52fbfed94837ef6ab5e0cca61ac218bb82d01b0e4314fb10264a9ef
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} {% if config.params.extra.run_generation %}
  python -m profbench.run_report_generation \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --folder {{config.output_dir}}{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.extra.version is not none %} --version {{config.params.extra.version}}{% endif %}{% if config.params.extra.upload_documents %} --upload-documents{% endif %}{% if config.params.extra.web_search %} --web-search{% endif %}{% if config.params.extra.reasoning %} --reasoning{% endif %}{% if config.params.extra.reasoning_effort is not none %} --reasoning-effort {{config.params.extra.reasoning_effort}}{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  GENERATION_OUTPUT=$(ls -t {{config.output_dir}}/*.jsonl | head -1) && 
{% endif %} {% if config.params.extra.run_judge_generated %}
  python -m profbench.run_best_llm_judge_on_generated_reports \
    --filename $GENERATION_OUTPUT \
    --api-key $API_KEY \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --output-folder {{config.output_dir}}/judgements{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  JUDGE_OUTPUT=$(ls -t {{config.output_dir}}/judgements/*.jsonl | head -1) && 
  python -m profbench.score_report_generation $JUDGE_OUTPUT
{% endif %} {% if config.params.extra.run_judge_provided %}
  python -m profbench.run_llm_judge_on_provided_reports \
    --model {{target.api_endpoint.model_id}} \
    --library {{config.params.extra.library}} \
    --timeout {{config.params.request_timeout}} \
    --parallel {{config.params.parallelism}} \
    --retry-attempts {{config.params.max_retries}} \
    --folder {{config.output_dir}}{% if target.api_endpoint.url is not none %} --base-url {{target.api_endpoint.url}}{% endif %}{% if config.params.extra.reasoning %} --reasoning{% endif %}{% if config.params.extra.reasoning_effort is not none %} --reasoning-effort {{config.params.extra.reasoning_effort}}{% endif %}{% if config.params.extra.debug %} --debug{% endif %}{% if config.params.limit_samples is not none %} --limit-samples {{config.params.limit_samples}}{% endif %}{% if config.params.temperature is not none %} --temperature {{config.params.temperature}}{% endif %}{% if config.params.top_p is not none %} --top-p {{config.params.top_p}}{% endif %}{% if config.params.max_new_tokens is not none %} --max-tokens {{config.params.max_new_tokens}}{% endif %} && 
  JUDGE_OUTPUT=$(ls -t {{config.output_dir}}/*.jsonl | head -1) && 
  python -m profbench.score_llm_judge $JUDGE_OUTPUT
{% endif %}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0.0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 600
    extra:
      run_generation: true
      run_judge_generated: true
      run_judge_provided: false
      library: openai
      version: lite
      upload_documents: false
      web_search: false
      reasoning: false
      reasoning_effort: null
      debug: false
  type: report_generation
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

