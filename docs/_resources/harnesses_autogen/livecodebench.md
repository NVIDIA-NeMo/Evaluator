# livecodebench

This page contains all evaluation tasks for the **livecodebench** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [AA_code_generation](#livecodebench-aa-code-generation)
  - AA code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.
* - [codeexecution_v2](#livecodebench-codeexecution-v2)
  - “Execute” a program on an input, evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.
* - [codeexecution_v2_cot](#livecodebench-codeexecution-v2-cot)
  - “CoT. Execute” a program on an input, evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.
* - [codegeneration_notfast](#livecodebench-codegeneration-notfast)
  - Not fast version of code generation (v2).
* - [codegeneration_release_latest](#livecodebench-codegeneration-release-latest)
  - Code generation latest version
* - [codegeneration_release_v1](#livecodebench-codegeneration-release-v1)
  - The initial release of the dataset with problems released between May 2023 and Mar 2024 containing 400 problems.
* - [codegeneration_release_v2](#livecodebench-codegeneration-release-v2)
  - The updated release of the dataset with problems released between May 2023 and May 2024 containing 511 problems.
* - [codegeneration_release_v3](#livecodebench-codegeneration-release-v3)
  - The updated release of the dataset with problems released between May 2023 and Jul 2024 containing 612 problems.
* - [codegeneration_release_v4](#livecodebench-codegeneration-release-v4)
  - The updated release of the dataset with problems released between May 2023 and Sep 2024 containing 713 problems.
* - [codegeneration_release_v5](#livecodebench-codegeneration-release-v5)
  - The updated release of the dataset with problems released between May 2023 and Jan 2025 containing 880 problems.
* - [codegeneration_release_v6](#livecodebench-codegeneration-release-v6)
  - The updated release of the dataset with problems released between May 2023 and Apr 2025 containing 1055 problems.
* - [livecodebench_0724_0125](#livecodebench-livecodebench-0724-0125)
  - - Code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result. - The data period and sampling parameters used by Artificial Analaysis (https://artificialanalysis.ai/methodology/intelligence-benchmarking)
* - [livecodebench_0824_0225](#livecodebench-livecodebench-0824-0225)
  - ['Code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.', 'The data period and sampling parameters used by NeMo Alignment team.']
* - [testoutputprediction](#livecodebench-testoutputprediction)
  - Solve the natural language task on a specified input, evaluating the ability to generate testing outputs. The model is given the natural language problem description and an input, and the output should be the output for the problem.
```

(livecodebench-aa-code-generation)=
## AA_code_generation

AA code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 3
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: 2024-07-01
      end_date: 2025-01-01
      cot_code_execution: false
      release_version: release_v5
    task: codegeneration
  type: AA_codegeneration
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codeexecution-v2)=
## codeexecution_v2

“Execute” a program on an input, evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v2
    task: codeexecution
  type: codeexecution_v2
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codeexecution-v2-cot)=
## codeexecution_v2_cot

“CoT. Execute” a program on an input, evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: true
      release_version: release_v2
    task: codeexecution
  type: codeexecution_v2_cot
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-notfast)=
## codegeneration_notfast

Not fast version of code generation (v2).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      args: --not_fast
    task: codegeneration
  type: codegeneration_notfast
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-latest)=
## codegeneration_release_latest

Code generation latest version

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_latest
    task: codegeneration
  type: codegeneration_release_latest
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v1)=
## codegeneration_release_v1

The initial release of the dataset with problems released between May 2023 and Mar 2024 containing 400 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v1
    task: codegeneration
  type: codegeneration_release_v1
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v2)=
## codegeneration_release_v2

The updated release of the dataset with problems released between May 2023 and May 2024 containing 511 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v2
    task: codegeneration
  type: codegeneration_release_v2
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v3)=
## codegeneration_release_v3

The updated release of the dataset with problems released between May 2023 and Jul 2024 containing 612 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v3
    task: codegeneration
  type: codegeneration_release_v3
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v4)=
## codegeneration_release_v4

The updated release of the dataset with problems released between May 2023 and Sep 2024 containing 713 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v4
    task: codegeneration
  type: codegeneration_release_v4
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v5)=
## codegeneration_release_v5

The updated release of the dataset with problems released between May 2023 and Jan 2025 containing 880 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v5
    task: codegeneration
  type: codegeneration_release_v5
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-codegeneration-release-v6)=
## codegeneration_release_v6

The updated release of the dataset with problems released between May 2023 and Apr 2025 containing 1055 problems.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_v6
    task: codegeneration
  type: codegeneration_release_v6
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-livecodebench-0724-0125)=
## livecodebench_0724_0125

- Code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result. - The data period and sampling parameters used by Artificial Analaysis (https://artificialanalysis.ai/methodology/intelligence-benchmarking)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 3
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: 2024-07-01
      end_date: 2025-01-01
      cot_code_execution: false
      release_version: release_v5
    task: codegeneration
  type: livecodebench_0724_0125
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-livecodebench-0824-0225)=
## livecodebench_0824_0225

['Code generation evaluating code comprehension ability. The model is given a program and an input, and the output should be the result.', 'The data period and sampling parameters used by NeMo Alignment team.']

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 3
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: 2024-08-01
      end_date: 2025-02-01
      cot_code_execution: false
      release_version: release_v5
    task: codegeneration
  type: livecodebench_0824_0225
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(livecodebench-testoutputprediction)=
## testoutputprediction

Solve the natural language task on a specified input, evaluating the ability to generate testing outputs. The model is given the natural language problem description and an input, and the output should be the output for the problem.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `livecodebench`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/livecodebench:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:98a3aa0b4b09052e54f9ba4c689b75aab501230e4b3c9700ae244a457e960817
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}
  export API_KEY=${{target.api_endpoint.api_key}} && 
{% endif %} livecodebench --model {{target.api_endpoint.model_id}} \
            --scenario {{config.params.task}} \
            --release_version {{config.params.extra.release_version}} \
            --url {{target.api_endpoint.url}} \
            --temperature {{config.params.temperature}} \
            --top_p {{config.params.top_p}} \
            --evaluate \
            --codegen_n {{config.params.extra.n_samples}} \
            --use_cache \
            --cache_batch_size {{config.params.extra.cache_batch_size}} \
            --num_process_evaluate {{config.params.extra.num_process_evaluate}} \
            --n {{config.params.extra.n_samples}} \
            --max_tokens {{config.params.max_new_tokens}} \
            --out_dir {{config.output_dir}} \
            --multiprocess {{config.params.parallelism}} \
            --max_retries {{config.params.max_retries}} \
            --timeout {{config.params.request_timeout}}{% if config.params.extra.start_date is not none %} --start_date {{config.params.extra.start_date}} {% endif %} {% if config.params.extra.end_date is not none %} --end_date {{config.params.extra.end_date}} {% endif %} {% if config.params.extra.support_system_role %} --support_system_role {% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %}{% if config.params.extra.cot_code_execution == true %} --cot_code_execution {% endif %}{% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %}

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
    request_timeout: 60
    extra:
      n_samples: 10
      num_process_evaluate: 5
      cache_batch_size: 10
      support_system_role: false
      start_date: null
      end_date: null
      cot_code_execution: false
      release_version: release_latest
    task: testoutputprediction
  type: testoutputprediction
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

