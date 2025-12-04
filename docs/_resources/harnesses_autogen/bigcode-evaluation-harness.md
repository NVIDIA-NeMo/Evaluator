# bigcode-evaluation-harness

This page contains all evaluation tasks for the **bigcode-evaluation-harness** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [humaneval](#bigcode-evaluation-harness-humaneval)
  - HumanEval is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.
* - [humaneval_instruct](#bigcode-evaluation-harness-humaneval-instruct)
  - InstructHumanEval is a modified version of OpenAI HumanEval. For a given prompt, we extracted its signature, its docstring as well as its header to create a flexing setting which would allow to evaluation instruction-tuned LLM. The delimiters used in the instruction-tuning procedure can be use to build and instruction that would allow the model to elicit its best capabilities.
* - [humanevalplus](#bigcode-evaluation-harness-humanevalplus)
  - HumanEvalPlus is a modified version of HumanEval containing 80x more test cases.
* - [mbpp](#bigcode-evaluation-harness-mbpp)
  - MBPP consists of Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.
* - [mbppplus](#bigcode-evaluation-harness-mbppplus)
  - MBPP+ is a modified version of MBPP containing 35x more test cases.
* - [mbppplus_nemo](#bigcode-evaluation-harness-mbppplus-nemo)
  - MBPP+NeMo is a modified version of MBPP+ that uses the NeMo alignment prompt template.
* - [multiple-cpp](#bigcode-evaluation-harness-multiple-cpp)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-cs](#bigcode-evaluation-harness-multiple-cs)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-d](#bigcode-evaluation-harness-multiple-d)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-go](#bigcode-evaluation-harness-multiple-go)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-java](#bigcode-evaluation-harness-multiple-java)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-jl](#bigcode-evaluation-harness-multiple-jl)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-js](#bigcode-evaluation-harness-multiple-js)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-lua](#bigcode-evaluation-harness-multiple-lua)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-php](#bigcode-evaluation-harness-multiple-php)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-pl](#bigcode-evaluation-harness-multiple-pl)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-py](#bigcode-evaluation-harness-multiple-py)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-r](#bigcode-evaluation-harness-multiple-r)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-rb](#bigcode-evaluation-harness-multiple-rb)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-rkt](#bigcode-evaluation-harness-multiple-rkt)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-rs](#bigcode-evaluation-harness-multiple-rs)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-scala](#bigcode-evaluation-harness-multiple-scala)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-sh](#bigcode-evaluation-harness-multiple-sh)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-swift](#bigcode-evaluation-harness-multiple-swift)
  - MultiPL-E is a suite of coding tasks for many programming languages
* - [multiple-ts](#bigcode-evaluation-harness-multiple-ts)
  - MultiPL-E is a suite of coding tasks for many programming languages
```

(bigcode-evaluation-harness-humaneval)=
## humaneval

HumanEval is used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `humaneval`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: humaneval
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 20
  supported_endpoint_types:
  - completions
  type: humaneval
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-humaneval-instruct)=
## humaneval_instruct

InstructHumanEval is a modified version of OpenAI HumanEval. For a given prompt, we extracted its signature, its docstring as well as its header to create a flexing setting which would allow to evaluation instruction-tuned LLM. The delimiters used in the instruction-tuning procedure can be use to build and instruction that would allow the model to elicit its best capabilities.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `humaneval_instruct`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: instruct-humaneval-nocontext-py
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 20
  supported_endpoint_types:
  - chat
  type: humaneval_instruct
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-humanevalplus)=
## humanevalplus

HumanEvalPlus is a modified version of HumanEval containing 80x more test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `humanevalplus`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: humanevalplus
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: humanevalplus
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-mbpp)=
## mbpp

MBPP consists of Python programming problems, designed to be solvable by entry level programmers, covering programming fundamentals, standard library functionality, and so on. Each problem consists of a task description, code solution and 3 automated test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `mbpp`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: mbpp
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 10
  supported_endpoint_types:
  - completions
  - chat
  type: mbpp
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-mbppplus)=
## mbppplus

MBPP+ is a modified version of MBPP containing 35x more test cases.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `mbppplus`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: mbppplus
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  - chat
  type: mbppplus
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-mbppplus-nemo)=
## mbppplus_nemo

MBPP+NeMo is a modified version of MBPP+ that uses the NeMo alignment prompt template.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `mbppplus_nemo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 10
    task: mbppplus_nemo
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - chat
  type: mbppplus_nemo
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-cpp)=
## multiple-cpp

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-cpp`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-cpp
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-cpp
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-cs)=
## multiple-cs

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-cs`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-cs
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-cs
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-d)=
## multiple-d

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-d`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-d
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-d
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-go)=
## multiple-go

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-go`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-go
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-go
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-java)=
## multiple-java

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-java`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-java
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-java
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-jl)=
## multiple-jl

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-jl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-jl
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-jl
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-js)=
## multiple-js

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-js`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-js
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-js
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-lua)=
## multiple-lua

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-lua`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-lua
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-lua
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-php)=
## multiple-php

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-php`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-php
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-php
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-pl)=
## multiple-pl

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-pl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-pl
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-pl
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-py)=
## multiple-py

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-py`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-py
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-py
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-r)=
## multiple-r

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-r`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-r
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-r
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-rb)=
## multiple-rb

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-rb`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-rb
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-rb
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-rkt)=
## multiple-rkt

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-rkt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-rkt
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-rkt
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-rs)=
## multiple-rs

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-rs`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-rs
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-rs
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-scala)=
## multiple-scala

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-scala`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-scala
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-scala
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-sh)=
## multiple-sh

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-sh`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-sh
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-sh
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-swift)=
## multiple-swift

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-swift`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-swift
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-swift
target:
  api_endpoint: {}
```

</details>


(bigcode-evaluation-harness-multiple-ts)=
## multiple-ts

MultiPL-E is a suite of coding tasks for many programming languages

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bigcode-evaluation-harness`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bigcode-evaluation-harness:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:48f9e824c3992aa9043d2c2a1cde78f21cffd0a8273d78c6e676fa90891b1ab1
```

**Task Type:** `multiple-ts`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}NVCF_TOKEN=${{target.api_endpoint.api_key}}{% endif %} bigcode-eval --model_type {% if target.api_endpoint.type == "completions" %}nim-base{% elif target.api_endpoint.type == "chat" %}nim-chat{% endif %} --url {{target.api_endpoint.url}} --model_kwargs '{"model_name": "{{target.api_endpoint.model_id}}", "timeout": {{config.params.request_timeout}}, "connection_retries": {{config.params.max_retries}}}' --out_dir {{config.output_dir}} --task {{config.params.task}} --allow_code_execution --n_samples={{config.params.extra.n_samples}} {% if config.params.limit_samples is not none %}--limit {{config.params.limit_samples}}{% endif %} --max_new_tokens={{config.params.max_new_tokens}} --do_sample={{config.params.extra.do_sample}} --top_p {{config.params.top_p}} --temperature {{config.params.temperature}} --async_limit {{config.params.parallelism}}{% if config.params.extra.args is defined %} {{config.params.extra.args}} {% endif %}
```

**Defaults:**
```yaml
framework_name: bigcode-evaluation-harness
pkg_name: bigcode_evaluation_harness
config:
  params:
    max_new_tokens: 1024
    max_retries: 5
    parallelism: 10
    task: multiple-ts
    temperature: 0.1
    request_timeout: 30
    top_p: 0.95
    extra:
      do_sample: true
      n_samples: 5
  supported_endpoint_types:
  - completions
  type: multiple-ts
target:
  api_endpoint: {}
```

</details>

