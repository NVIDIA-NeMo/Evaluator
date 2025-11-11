# helm

This page contains all evaluation tasks for the **helm** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [aci_bench](#helm-aci-bench)
  - Extract and structure information from patient-doctor conversations
* - [ehr_sql](#helm-ehr-sql)
  - Given a natural language instruction, generate an SQL query that would be used in clinical research.
* - [head_qa](#helm-head-qa)
  - A collection of biomedical multiple-choice questions for testing medical knowledge (Vilares et al., 2019).
* - [med_dialog_healthcaremagic](#helm-med-dialog-healthcaremagic)
  - Generate summaries of doctor-patient conversations, healthcaremagic version
* - [med_dialog_icliniq](#helm-med-dialog-icliniq)
  - Generate summaries of doctor-patient conversations, icliniq version
* - [medbullets](#helm-medbullets)
  - A USMLE-style medical question dataset with multiple-choice answers and explanations (MedBullets, 2025).
* - [medcalc_bench](#helm-medcalc-bench)
  - A dataset which consists of a patient note, a question requesting to compute a specific medical value, and a ground truth answer (Khandekar et al., 2024).
* - [medec](#helm-medec)
  - A dataset containing medical narratives with error detection and correction pairs (Abacha et al., 2025).
* - [medhallu](#helm-medhallu)
  - A dataset of PubMed articles and associated questions, with the objective being to classify whether the answer is factual or hallucinated.
* - [medi_qa](#helm-medi-qa)
  - Retrieve and rank answers based on medical question understanding
* - [medication_qa](#helm-medication-qa)
  - Answer consumer medication-related questions
* - [mtsamples_procedures](#helm-mtsamples-procedures)
  - Document and extract information about medical procedures
* - [mtsamples_replicate](#helm-mtsamples-replicate)
  - Generate treatment plans based on clinical notes
* - [pubmed_qa](#helm-pubmed-qa)
  - A dataset that provides pubmed abstracts and asks associated questions yes/no/maybe questions.
* - [race_based_med](#helm-race-based-med)
  - A collection of LLM outputs in response to medical questions with race-based biases, with the objective being to classify whether the output contains racially biased content.
```

(helm-aci-bench)=
## aci_bench

Extract and structure information from patient-doctor conversations

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: aci_bench
  type: aci_bench
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-ehr-sql)=
## ehr_sql

Given a natural language instruction, generate an SQL query that would be used in clinical research.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: ehr_sql
  type: ehr_sql
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-head-qa)=
## head_qa

A collection of biomedical multiple-choice questions for testing medical knowledge (Vilares et al., 2019).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: head_qa
  type: head_qa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-med-dialog-healthcaremagic)=
## med_dialog_healthcaremagic

Generate summaries of doctor-patient conversations, healthcaremagic version

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: healthcaremagic
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: med_dialog
  type: med_dialog_healthcaremagic
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-med-dialog-icliniq)=
## med_dialog_icliniq

Generate summaries of doctor-patient conversations, icliniq version

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: icliniq
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: med_dialog
  type: med_dialog_icliniq
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medbullets)=
## medbullets

A USMLE-style medical question dataset with multiple-choice answers and explanations (MedBullets, 2025).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medbullets
  type: medbullets
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medcalc-bench)=
## medcalc_bench

A dataset which consists of a patient note, a question requesting to compute a specific medical value, and a ground truth answer (Khandekar et al., 2024).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medcalc_bench
  type: medcalc_bench
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medec)=
## medec

A dataset containing medical narratives with error detection and correction pairs (Abacha et al., 2025).

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medec
  type: medec
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medhallu)=
## medhallu

A dataset of PubMed articles and associated questions, with the objective being to classify whether the answer is factual or hallucinated.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medhallu
  type: medhallu
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medi-qa)=
## medi_qa

Retrieve and rank answers based on medical question understanding

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medi_qa
  type: medi_qa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-medication-qa)=
## medication_qa

Answer consumer medication-related questions

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: medication_qa
  type: medication_qa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-mtsamples-procedures)=
## mtsamples_procedures

Document and extract information about medical procedures

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: mtsamples_procedures
  type: mtsamples_procedures
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-mtsamples-replicate)=
## mtsamples_replicate

Generate treatment plans based on clinical notes

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: mtsamples_replicate
  type: mtsamples_replicate
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-pubmed-qa)=
## pubmed_qa

A dataset that provides pubmed abstracts and asks associated questions yes/no/maybe questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: pubmed_qa
  type: pubmed_qa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(helm-race-based-med)=
## race_based_med

A collection of LLM outputs in response to medical questions with race-based biases, with the objective being to classify whether the output contains racially biased content.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `helm`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/helm:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:6d7b6151100405d700c97d55b9233d98f18a013de357338321ca1a0b14999496
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export OPENAI_API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.gpt_judge_api_key is not none %}export GPT_JUDGE_API_KEY=${{config.params.extra.gpt_judge_api_key}} && {% endif %} {% if config.params.extra.llama_judge_api_key is not none %}export LLAMA_JUDGE_API_KEY=${{config.params.extra.llama_judge_api_key}} && {% endif %} {% if config.params.extra.claude_judge_api_key is not none %}export CLAUDE_JUDGE_API_KEY=${{config.params.extra.claude_judge_api_key}} && {% endif %} helm-generate-dynamic-model-configs  --model-name {{target.api_endpoint.model_id}}  --base-url {{target.api_endpoint.url}}  --openai-model-name {{target.api_endpoint.model_id}}  --output-dir {{config.output_dir}} && helm-run  --run-entries {{config.params.task}}:{% if config.params.extra.subset is not none %}subset={{config.params.extra.subset}},{% endif %}model={{target.api_endpoint.model_id}}  {% if config.params.limit_samples is not none %} --max-eval-instances {{config.params.limit_samples}}  {% endif %} {% if config.params.parallelism is not none %} -n {{config.params.parallelism}}  {% endif %} --suite {{config.params.task}} {% if config.params.extra.num_train_trials is not none %}  --num-train-trials {{config.params.extra.num_train_trials}} {% endif %} {% if config.params.extra.data_path is not none %}  --data-path {{config.params.extra.data_path}} {% endif %} {% if config.params.extra.num_output_tokens is not none %}  --num-output-tokens {{config.params.extra.num_output_tokens}} {% endif %} {% if config.params.extra.subject is not none %}  --subject {{config.params.extra.subject}} {% endif %} {% if config.params.extra.condition is not none %}  --condition {{config.params.extra.condition}} {% endif %} {% if config.params.extra.max_length is not none %}  --max-length {{config.params.extra.max_length}} {% endif %}  -o {{config.output_dir}}  --local-path {{config.output_dir}}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    parallelism: 1
    extra:
      data_path: null
      num_output_tokens: null
      subject: null
      condition: null
      max_length: null
      num_train_trials: null
      subset: null
      gpt_judge_api_key: GPT_JUDGE_API_KEY
      llama_judge_api_key: LLAMA_JUDGE_API_KEY
      claude_judge_api_key: CLAUDE_JUDGE_API_KEY
    task: race_based_med
  type: race_based_med
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

