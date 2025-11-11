# vlmevalkit

This page contains all evaluation tasks for the **vlmevalkit** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [AI2D](#vlmevalkit-ai2d)
  - A benchmark for evaluating diagram understanding capabilities of large vision-language models.
* - [ChartQA](#vlmevalkit-chartqa)
  - A Benchmark for Question Answering about Charts with Visual and Logical Reasoning
* - [MMMU](#vlmevalkit-mmmu)
  - A benchmark for evaluating multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning.
* - [MathVista-MINI](#vlmevalkit-mathvista-mini)
  - Evaluating Math Reasoning in Visual Contexts
* - [OCR-Reasoning](#vlmevalkit-ocr-reasoning)
  - Comprehensive benchmark of 1,069 human-annotated examples designed to evaluate multimodal large language models on text-rich image reasoning tasks by assessing both final answers and the reasoning process across six core abilities and 18 practical tasks.
* - [OCRBench](#vlmevalkit-ocrbench)
  - Comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models
* - [SlideVQA](#vlmevalkit-slidevqa)
  - Evaluates ability to answer questions about slide decks by selecting relevant slides from multiple images
```

(vlmevalkit-ai2d)=
## AI2D

A benchmark for evaluating diagram understanding capabilities of large vision-language models.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: AI2D_TEST
        class: ImageMCQDataset
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  type: ai2d_judge
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-chartqa)=
## ChartQA

A Benchmark for Question Answering about Charts with Visual and Logical Reasoning

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: ChartQA_TEST
        class: ImageVQADataset
  type: chartqa
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-mmmu)=
## MMMU

A benchmark for evaluating multimodal models on massive multi-discipline tasks demanding college-level subject knowledge and deliberate reasoning.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: MMMU_DEV_VAL
        class: MMMUDataset
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  type: mmmu_judge
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-mathvista-mini)=
## MathVista-MINI

Evaluating Math Reasoning in Visual Contexts

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: MathVista_MINI
        class: MathVista
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  type: mathvista-mini
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-ocr-reasoning)=
## OCR-Reasoning

Comprehensive benchmark of 1,069 human-annotated examples designed to evaluate multimodal large language models on text-rich image reasoning tasks by assessing both final answers and the reasoning process across six core abilities and 18 practical tasks.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: OCR_Reasoning
        class: OCR_Reasoning
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  type: ocr_reasoning
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-ocrbench)=
## OCRBench

Comprehensive evaluation benchmark designed to assess the OCR capabilities of Large Multimodal Models

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: OCRBench
        class: OCRBench
  type: ocrbench
target:
  api_endpoint: {}

```

</details>


(vlmevalkit-slidevqa)=
## SlideVQA

Evaluates ability to answer questions about slide decks by selecting relevant slides from multiple images

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `vlmevalkit`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-vlm/vlmevalkit:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:54f9dcd86d62d9897e035a424a7b304dc636052f267afccb019af15eae93a4e0
```

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
config:
  supported_endpoint_types:
  - vlm
  params:
    limit_samples: null
    max_new_tokens: 2048
    temperature: 0
    top_p: null
    parallelism: 4
    max_retries: 5
    request_timeout: 60
    extra:
      dataset:
        name: SLIDEVQA
        class: SlideVQA
      judge:
        model: gpt-4o
        args: '{"use_azure": true}'
  type: slidevqa
target:
  api_endpoint: {}

```

</details>

