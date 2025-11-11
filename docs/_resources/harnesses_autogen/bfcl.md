# bfcl

This page contains all evaluation tasks for the **bfcl** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [bfclv2](#bfcl-bfclv2)
  - Single-turn, Live and Non-Live, AST and Exec evaluation
* - [bfclv2_ast](#bfcl-bfclv2-ast)
  - Single-turn, Live and Non-Live,  AST evaluation only. Uses native function calling.
* - [bfclv2_ast_prompting](#bfcl-bfclv2-ast-prompting)
  - Single-turn, Live and Non-Live,  AST evaluation only. Not using native function calling.
* - [bfclv3](#bfcl-bfclv3)
  - Single-turn and Multi-turn, Live and Non-Live, AST and Exec evaluation
* - [bfclv3_ast](#bfcl-bfclv3-ast)
  - Single-turn and Multi-turn, Live and Non-Live, AST evaluation. Uses native function calling.
* - [bfclv3_ast_prompting](#bfcl-bfclv3-ast-prompting)
  - Single-turn and Multi-turn, Live and Non-Live, AST evaluation. Not using native function calling.
```

(bfcl-bfclv2)=
## bfclv2

Single-turn, Live and Non-Live, AST and Exec evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: single_turn
  type: bfclv2
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>


(bfcl-bfclv2-ast)=
## bfclv2_ast

Single-turn, Live and Non-Live,  AST evaluation only. Uses native function calling.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: true
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: ast
  type: bfclv2_ast
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>


(bfcl-bfclv2-ast-prompting)=
## bfclv2_ast_prompting

Single-turn, Live and Non-Live,  AST evaluation only. Not using native function calling.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: ast
  type: bfclv2_ast_prompting
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>


(bfcl-bfclv3)=
## bfclv3

Single-turn and Multi-turn, Live and Non-Live, AST and Exec evaluation

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: all
  type: bfclv3
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>


(bfcl-bfclv3-ast)=
## bfclv3_ast

Single-turn and Multi-turn, Live and Non-Live, AST evaluation. Uses native function calling.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: true
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: multi_turn,ast
  type: bfclv3_ast
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>


(bfcl-bfclv3-ast-prompting)=
## bfclv3_ast_prompting

Single-turn and Multi-turn, Live and Non-Live, AST evaluation. Not using native function calling.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `bfcl`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:afd09c97a14d7a97e63cb3e765b8738a8c5d53ecc1771c953b6283f75f066dfa
```

**Command:**
```bash
{%- if config.params.extra.custom_dataset.path is not none and config.params.extra.custom_dataset.format is not none -%} echo "Processing custom dataset..." && export BFCL_DATA_DIR=$(core-evals-process-custom-dataset \
  --dataset_format {{config.params.extra.custom_dataset.format}} \
  --dataset_path {{config.params.extra.custom_dataset.path}} \
  --test_category {{config.params.task}} \
  --processing_output_dir {{config.output_dir ~ "/custom_dataset_processing"}} \
  {% if config.params.extra.custom_dataset.data_template_path %}--data_template_path {{config.params.extra.custom_dataset.data_template_path}}{% endif %}) && \
echo "Using custom dataset at ${BFCL_DATA_DIR}" && \
{% endif -%}
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl generate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}  {% if config.params.limit_samples is not none %} --limit {{config.params.limit_samples}}{% endif %} --num-threads  {{config.params.parallelism}} && \
{% if target.api_endpoint.api_key is not none %}OPENAI_API_KEY=${{target.api_endpoint.api_key}}{% endif %} bfcl evaluate --model {{target.api_endpoint.model_id}} --test-category {{config.params.task}} --model-mapping oai --result-dir {{config.output_dir}} --score-dir {{config.output_dir}} --model-args base_url={{target.api_endpoint.url}},native_calling={{config.params.extra.native_calling}}

```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 10
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
    task: multi_turn,ast
  type: bfclv3_ast_prompting
  supported_endpoint_types:
  - chat
  - vlm
target:
  api_endpoint: {}

```

</details>

