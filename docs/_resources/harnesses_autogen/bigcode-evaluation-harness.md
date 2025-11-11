# bigcode-evaluation-harness

This page contains all evaluation tasks for the **bigcode-evaluation-harness** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [HumanEval](#bigcode-evaluation-harness-humaneval)
  - HumanEval is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.
* - [HumanEval+](#bigcode-evaluation-harness-humaneval)
  - HumanEvalPlus is a modified version of HumanEval containing 80x more test cases.
* - [HumanEval-Instruct](#bigcode-evaluation-harness-humaneval-instruct)
  - InstructHumanEval is a modified version of OpenAI HumanEval. For a given prompt, we extracted its signature, its docstring as well as its header to create a flexing setting which would allow to evaluation instruction-tuned LLM. The delimiters used in the instruction-tuning procedure can be use to build and instruction that would allow the model to elicit its best capabilities.
* - [MBPP](#bigcode-evaluation-harness-mbpp)
  - MBPP consists of Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.
* - [MBPP+](#bigcode-evaluation-harness-mbpp)
  - MBPP+ is a modified version of MBPP containing 35x more test cases.
* - [MBPP+NeMo](#bigcode-evaluation-harness-mbppnemo)
  - MBPP+NeMo is a modified version of MBPP+ that uses the NeMo alignment prompt template.
* - [MultiPL-E-cpp](#bigcode-evaluation-harness-multipl-e-cpp)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-cs](#bigcode-evaluation-harness-multipl-e-cs)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-d](#bigcode-evaluation-harness-multipl-e-d)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-go](#bigcode-evaluation-harness-multipl-e-go)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-java](#bigcode-evaluation-harness-multipl-e-java)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-jl](#bigcode-evaluation-harness-multipl-e-jl)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-js](#bigcode-evaluation-harness-multipl-e-js)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-lua](#bigcode-evaluation-harness-multipl-e-lua)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-php](#bigcode-evaluation-harness-multipl-e-php)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-pl](#bigcode-evaluation-harness-multipl-e-pl)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-py](#bigcode-evaluation-harness-multipl-e-py)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-r](#bigcode-evaluation-harness-multipl-e-r)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-rb](#bigcode-evaluation-harness-multipl-e-rb)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-rkt](#bigcode-evaluation-harness-multipl-e-rkt)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-rs](#bigcode-evaluation-harness-multipl-e-rs)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-scala](#bigcode-evaluation-harness-multipl-e-scala)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-sh](#bigcode-evaluation-harness-multipl-e-sh)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-swift](#bigcode-evaluation-harness-multipl-e-swift)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [MultiPL-E-ts](#bigcode-evaluation-harness-multipl-e-ts)
  - MultiPL-E is a suite of coding tasks for many programming languages
```

(bigcode-evaluation-harness-humaneval)=
## HumanEval

HumanEval is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 20
    task: humaneval
  type: humaneval
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-humaneval)=
## HumanEval+

HumanEvalPlus is a modified version of HumanEval containing 80x more test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: humanevalplus
  type: humanevalplus
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-humaneval-instruct)=
## HumanEval-Instruct

InstructHumanEval is a modified version of OpenAI HumanEval. For a given prompt, we extracted its signature, its docstring as well as its header to create a flexing setting which would allow to evaluation instruction-tuned LLM. The delimiters used in the instruction-tuning procedure can be use to build and instruction that would allow the model to elicit its best capabilities.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 20
    task: instruct-humaneval-nocontext-py
  type: humaneval_instruct
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-mbpp)=
## MBPP

MBPP consists of Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  - chat
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 10
    task: mbpp
  type: mbpp
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-mbpp)=
## MBPP+

MBPP+ is a modified version of MBPP containing 35x more test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  - chat
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: mbppplus
  type: mbppplus
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-mbppnemo)=
## MBPP+NeMo

MBPP+NeMo is a modified version of MBPP+ that uses the NeMo alignment prompt template.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - chat
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: mbppplus_nemo
  type: mbppplus_nemo
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-cpp)=
## MultiPL-E-cpp

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-cpp
  type: multiple-cpp
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-cs)=
## MultiPL-E-cs

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-cs
  type: multiple-cs
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-d)=
## MultiPL-E-d

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-d
  type: multiple-d
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-go)=
## MultiPL-E-go

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-go
  type: multiple-go
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-java)=
## MultiPL-E-java

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-java
  type: multiple-java
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-jl)=
## MultiPL-E-jl

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-jl
  type: multiple-jl
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-js)=
## MultiPL-E-js

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-js
  type: multiple-js
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-lua)=
## MultiPL-E-lua

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-lua
  type: multiple-lua
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-php)=
## MultiPL-E-php

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-php
  type: multiple-php
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-pl)=
## MultiPL-E-pl

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-pl
  type: multiple-pl
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-py)=
## MultiPL-E-py

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-py
  type: multiple-py
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-r)=
## MultiPL-E-r

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-r
  type: multiple-r
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-rb)=
## MultiPL-E-rb

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-rb
  type: multiple-rb
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-rkt)=
## MultiPL-E-rkt

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-rkt
  type: multiple-rkt
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-rs)=
## MultiPL-E-rs

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-rs
  type: multiple-rs
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-scala)=
## MultiPL-E-scala

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-scala
  type: multiple-scala
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-sh)=
## MultiPL-E-sh

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-sh
  type: multiple-sh
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-swift)=
## MultiPL-E-swift

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-swift
  type: multiple-swift
target:
  api_endpoint: {}

```

</details>


(bigcode-evaluation-harness-multipl-e-ts)=
## MultiPL-E-ts

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bfe529f539895b0a1b1e5b4d13f208c8345e291e59220f54e2434a8c3467b3b5
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
config:
  supported_endpoint_types:
  - completions
  params:
    limit_samples: null
    max_new_tokens: 1024
    temperature: 0.1
    top_p: 0.95
    parallelism: 10
    max_retries: 5
    request_timeout: 30
    extra:
      do_sample: true
      n_samples: 5
    task: multiple-ts
  type: multiple-ts
target:
  api_endpoint: {}

```

</details>

