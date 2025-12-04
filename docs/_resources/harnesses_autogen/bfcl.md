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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv2`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: single_turn
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv2_ast`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: ast
    extra:
      native_calling: true
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv2_ast
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv2_ast_prompting`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: ast
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv2_ast_prompting
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv3`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: all
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv3
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv3_ast`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: multi_turn,ast
    extra:
      native_calling: true
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv3_ast
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/bfcl:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:b25ff910708c61de2588440306ff2fe331521ad84e18525b39f3184baa32512b
```

**Task Type:** `bfclv3_ast_prompting`

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
framework_name: bfcl
pkg_name: bfcl
config:
  params:
    parallelism: 10
    task: multi_turn,ast
    extra:
      native_calling: false
      custom_dataset:
        path: null
        format: null
        data_template_path: null
  supported_endpoint_types:
  - chat
  - vlm
  type: bfclv3_ast_prompting
target:
  api_endpoint: {}
```

</details>

