# scicode

This page contains all evaluation tasks for the **scicode** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [aa_scicode](#scicode-aa-scicode)
  - - SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant mimicks setup used by Artificial Analysis in their Intelligence Benchmark (v2). - It includes scientist-annotated background in the prompts and uses all available problems for evaluation (including "dev" set).
* - [scicode](#scicode-scicode)
  - - SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant does not include scientist-annotated background in the prompts.
* - [scicode_aa_v2](#scicode-scicode-aa-v2)
  - - SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant mimicks setup used by Artificial Analysis in their Intelligence Benchmark (v2). - It includes scientist-annotated background in the prompts and uses all available problems for evaluation (including "dev" set).
* - [scicode_background](#scicode-scicode-background)
  - - SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant includes scientist-annotated background in the prompts.
```

(scicode-aa-scicode)=
## aa_scicode

- SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant mimicks setup used by Artificial Analysis in their Intelligence Benchmark (v2). - It includes scientist-annotated background in the prompts and uses all available problems for evaluation (including "dev" set).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `scicode`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/scicode:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:a2bc94be8496bf9c1bfc31e02e25709503092f6c9b7ad4beae76a56c19b7c594
```

**Task Type:** `aa_scicode`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}API_KEY=${{target.api_endpoint.api_key}}{% endif %} scicode_eval --model {{target.api_endpoint.model_id}} --url {{target.api_endpoint.url}} --output-dir {{config.output_dir}} --log-dir {{config.output_dir}}/logs {% if config.params.temperature is not none %}--temperature={{config.params.temperature}}{% endif %} {% if config.params.limit_samples is not none %}--limit-samples={{config.params.limit_samples}}{% endif %} --n-samples={{config.params.extra.n_samples}} --extra-params top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},max_tokens={{config.params.max_new_tokens}},max_retries={{config.params.max_retries}} {% if config.params.extra.with_background %}--with-background {% endif %} {% if config.params.extra.include_dev %}--include-dev{% endif %} {% if config.params.extra.eval_threads is not none %}--eval-threads={{config.params.extra.eval_threads}}{% endif %} --concurrent-requests={{config.params.parallelism}}
```

**Defaults:**
```yaml
framework_name: scicode
pkg_name: scicode
config:
  params:
    max_new_tokens: 4096
    max_retries: 2
    parallelism: 1
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
    extra:
      with_background: true
      include_dev: true
      n_samples: 3
      eval_threads: null
  supported_endpoint_types:
  - chat
  type: aa_scicode
target:
  api_endpoint:
    stream: false
```

</details>


(scicode-scicode)=
## scicode

- SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant does not include scientist-annotated background in the prompts.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `scicode`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/scicode:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:a2bc94be8496bf9c1bfc31e02e25709503092f6c9b7ad4beae76a56c19b7c594
```

**Task Type:** `scicode`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}API_KEY=${{target.api_endpoint.api_key}}{% endif %} scicode_eval --model {{target.api_endpoint.model_id}} --url {{target.api_endpoint.url}} --output-dir {{config.output_dir}} --log-dir {{config.output_dir}}/logs {% if config.params.temperature is not none %}--temperature={{config.params.temperature}}{% endif %} {% if config.params.limit_samples is not none %}--limit-samples={{config.params.limit_samples}}{% endif %} --n-samples={{config.params.extra.n_samples}} --extra-params top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},max_tokens={{config.params.max_new_tokens}},max_retries={{config.params.max_retries}} {% if config.params.extra.with_background %}--with-background {% endif %} {% if config.params.extra.include_dev %}--include-dev{% endif %} {% if config.params.extra.eval_threads is not none %}--eval-threads={{config.params.extra.eval_threads}}{% endif %} --concurrent-requests={{config.params.parallelism}}
```

**Defaults:**
```yaml
framework_name: scicode
pkg_name: scicode
config:
  params:
    max_new_tokens: 2048
    max_retries: 2
    parallelism: 1
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
    extra:
      with_background: false
      include_dev: false
      n_samples: 1
      eval_threads: null
  supported_endpoint_types:
  - chat
  type: scicode
target:
  api_endpoint:
    stream: false
```

</details>


(scicode-scicode-aa-v2)=
## scicode_aa_v2

- SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant mimicks setup used by Artificial Analysis in their Intelligence Benchmark (v2). - It includes scientist-annotated background in the prompts and uses all available problems for evaluation (including "dev" set).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `scicode`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/scicode:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:a2bc94be8496bf9c1bfc31e02e25709503092f6c9b7ad4beae76a56c19b7c594
```

**Task Type:** `scicode_aa_v2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}API_KEY=${{target.api_endpoint.api_key}}{% endif %} scicode_eval --model {{target.api_endpoint.model_id}} --url {{target.api_endpoint.url}} --output-dir {{config.output_dir}} --log-dir {{config.output_dir}}/logs {% if config.params.temperature is not none %}--temperature={{config.params.temperature}}{% endif %} {% if config.params.limit_samples is not none %}--limit-samples={{config.params.limit_samples}}{% endif %} --n-samples={{config.params.extra.n_samples}} --extra-params top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},max_tokens={{config.params.max_new_tokens}},max_retries={{config.params.max_retries}} {% if config.params.extra.with_background %}--with-background {% endif %} {% if config.params.extra.include_dev %}--include-dev{% endif %} {% if config.params.extra.eval_threads is not none %}--eval-threads={{config.params.extra.eval_threads}}{% endif %} --concurrent-requests={{config.params.parallelism}}
```

**Defaults:**
```yaml
framework_name: scicode
pkg_name: scicode
config:
  params:
    max_new_tokens: 16384
    max_retries: 30
    parallelism: 1
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
    extra:
      with_background: true
      include_dev: true
      n_samples: 3
      eval_threads: null
  supported_endpoint_types:
  - chat
  type: scicode_aa_v2
target:
  api_endpoint:
    stream: false
```

</details>


(scicode-scicode-background)=
## scicode_background

- SciCode is a challenging benchmark designed to evaluate the capabilities of LLMs in generating code for solving realistic scientific research problems. - This variant includes scientist-annotated background in the prompts.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `scicode`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/scicode:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:a2bc94be8496bf9c1bfc31e02e25709503092f6c9b7ad4beae76a56c19b7c594
```

**Task Type:** `scicode_background`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}API_KEY=${{target.api_endpoint.api_key}}{% endif %} scicode_eval --model {{target.api_endpoint.model_id}} --url {{target.api_endpoint.url}} --output-dir {{config.output_dir}} --log-dir {{config.output_dir}}/logs {% if config.params.temperature is not none %}--temperature={{config.params.temperature}}{% endif %} {% if config.params.limit_samples is not none %}--limit-samples={{config.params.limit_samples}}{% endif %} --n-samples={{config.params.extra.n_samples}} --extra-params top_p={{config.params.top_p}},timeout={{config.params.request_timeout}},max_tokens={{config.params.max_new_tokens}},max_retries={{config.params.max_retries}} {% if config.params.extra.with_background %}--with-background {% endif %} {% if config.params.extra.include_dev %}--include-dev{% endif %} {% if config.params.extra.eval_threads is not none %}--eval-threads={{config.params.extra.eval_threads}}{% endif %} --concurrent-requests={{config.params.parallelism}}
```

**Defaults:**
```yaml
framework_name: scicode
pkg_name: scicode
config:
  params:
    max_new_tokens: 2048
    max_retries: 2
    parallelism: 1
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
    extra:
      with_background: true
      include_dev: false
      n_samples: 1
      eval_threads: null
  supported_endpoint_types:
  - chat
  type: scicode_background
target:
  api_endpoint:
    stream: false
```

</details>

