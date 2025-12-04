# vlmevalkit

This page contains all evaluation tasks for the **vlmevalkit** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [ai2d_judge](#vlmevalkit-ai2d-judge)
  - A benchmark for evaluating diagram understanding capabilities of large vision-language models.
* - [chartqa](#vlmevalkit-chartqa)
  - A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
* - [mathvista-mini](#vlmevalkit-mathvista-mini)
  - Evaluating Math Reasoning in Visual Contexts
* - [mmmu_judge](#vlmevalkit-mmmu-judge)
  - A benchmark for evaluating multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning.
* - [ocr_reasoning](#vlmevalkit-ocr-reasoning)
  - Comprehensive benchmark of 1,069 human-annotated examples designed to evaluate multimodal large language models on text-rich image reasoning tasks by assessing both final answers and the reasoning process across six core abilities and 18 practical tasks.
* - [ocrbench](#vlmevalkit-ocrbench)
  - Comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models
* - [slidevqa](#vlmevalkit-slidevqa)
  - Evaluates ability to answer questions about slide decks by selecting relevant slides from multiple images
```

(vlmevalkit-ai2d-judge)=
## ai2d_judge

A benchmark for evaluating diagram understanding capabilities of large vision-language models.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `ai2d_judge`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: AI2D_TEST
        class: ImageMCQDataset
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  supported_endpoint_types:
  - vlm
  type: ai2d_judge
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-chartqa)=
## chartqa

A Benchmark for Question Answering about Charts with Visual and Logical Reasoning

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `chartqa`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: ChartQA_TEST
        class: ImageVQADataset
  supported_endpoint_types:
  - vlm
  type: chartqa
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-mathvista-mini)=
## mathvista-mini

Evaluating Math Reasoning in Visual Contexts

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `mathvista-mini`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: MathVista_MINI
        class: MathVista
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  supported_endpoint_types:
  - vlm
  type: mathvista-mini
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-mmmu-judge)=
## mmmu_judge

A benchmark for evaluating multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `mmmu_judge`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: MMMU_DEV_VAL
        class: MMMUDataset
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  supported_endpoint_types:
  - vlm
  type: mmmu_judge
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-ocr-reasoning)=
## ocr_reasoning

Comprehensive benchmark of 1,069 human-annotated examples designed to evaluate multimodal large language models on text-rich image reasoning tasks by assessing both final answers and the reasoning process across six core abilities and 18 practical tasks.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `ocr_reasoning`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: OCR_Reasoning
        class: OCR_Reasoning
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  supported_endpoint_types:
  - vlm
  type: ocr_reasoning
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-ocrbench)=
## ocrbench

Comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `ocrbench`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: OCRBench
        class: OCRBench
  supported_endpoint_types:
  - vlm
  type: ocrbench
target:
  api_endpoint: {}
```

</details>


(vlmevalkit-slidevqa)=
## slidevqa

Evaluates ability to answer questions about slide decks by selecting relevant slides from multiple images

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:bdb18b8879d25893d03bae7a2301f6bad2bdd20876e05c10b4ccd9b5ed20efb0
```

**Task Type:** `slidevqa`

**Command:**
```bash
cat > {{config.output_dir}}/vlmeval_config.json << 'EOF'
{
  "model": {
    "{{target.api_endpoint.model_id.split('/')[-1]}}": {
      "class": "CustomOAIEndpoint",
      "model": "{{target.api_endpoint.model_id}}",
      "api_base": "{{target.api_endpoint.url}}",
      "api_key_var_name": "{{target.api_endpoint.api_key}}",
      "max_tokens": {{config.params.max_new_tokens}},
      "temperature": {{config.params.temperature}},{% if config.params.top_p is not none %}
      "top_p": {{config.params.top_p}},{% endif %}
      "retry": {{config.params.max_retries}},
      "timeout": {{config.params.request_timeout}}{% if config.params.extra.wait is defined %},
      "wait": {{config.params.extra.wait}}{% endif %}{% if config.params.extra.img_size is defined %},
      "img_size": {{config.params.extra.img_size}}{% endif %}{% if config.params.extra.img_detail is defined %},
      "img_detail": "{{config.params.extra.img_detail}}"{% endif %}{% if config.params.extra.system_prompt is defined %},
      "system_prompt": "{{config.params.extra.system_prompt}}"{% endif %}{% if config.params.extra.verbose is defined %},
      "verbose": {{config.params.extra.verbose}}{% endif %}
    }
  },
  "data": {
    "{{config.params.extra.dataset.name}}": {
      "class": "{{config.params.extra.dataset.class}}",
      "dataset": "{{config.params.extra.dataset.name}}",
      "model": "{{target.api_endpoint.model_id}}"
    }
  }
}
EOF
python -m vlmeval.run \
  --config {{config.output_dir}}/vlmeval_config.json \
  --work-dir {{config.output_dir}} \
  --api-nproc {{config.params.parallelism}} \
  {%- if config.params.extra.judge is defined %}
  --judge {{config.params.extra.judge.model}} \
  --judge-args '{{config.params.extra.judge.args}}' \
  {%- endif %}
  {% if config.params.limit_samples is not none %}--first-n {{config.params.limit_samples}}{% endif %}

```

**Defaults:**
```yaml
framework_name: vlmevalkit
pkg_name: vlmevalkit
config:
  params:
    max_new_tokens: 2048
    max_retries: 5
    parallelism: 4
    temperature: 0.0
    request_timeout: 60
    extra:
      dataset:
        name: SLIDEVQA
        class: SlideVQA
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  supported_endpoint_types:
  - vlm
  type: slidevqa
target:
  api_endpoint: {}
```

</details>

