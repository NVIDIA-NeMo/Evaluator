# simple-evals

This page contains all evaluation tasks for the **simple-evals** harness.

```{list-table}
:header-rows: 1
:widths: 30 70

* - Task
  - Description
* - [AA_AIME_2024](#simple-evals-aa-aime-2024)
  - AA AIME 2024 questions, math
* - [AA_math_test_500](#simple-evals-aa-math-test-500)
  - AA Open Ai math test 500
* - [AIME_2024](#simple-evals-aime-2024)
  - AIME 2024 questions, math
* - [AIME_2025](#simple-evals-aime-2025)
  - AIME 2025 questions, math
* - [aime_2024_nemo](#simple-evals-aime-2024-nemo)
  - AIME 2024 questions, math, using NeMo's alignment template
* - [aime_2025_nemo](#simple-evals-aime-2025-nemo)
  - AIME 2025 questions, math, using NeMo's alignment template
* - [browsecomp](#simple-evals-browsecomp)
  - BrowseComp is a benchmark for measuring the ability for agents to browse the web.
* - [gpqa_diamond](#simple-evals-gpqa-diamond)
  - gpqa_diamond 0-shot CoT
* - [gpqa_diamond_aa_v2](#simple-evals-gpqa-diamond-aa-v2)
  - gpqa_diamond questions with custom regex extraction patterns for AA v2
* - [gpqa_diamond_aa_v2_llama_4](#simple-evals-gpqa-diamond-aa-v2-llama-4)
  - gpqa_diamond questions with custom regex extraction patterns for Llama 4
* - [gpqa_diamond_nemo](#simple-evals-gpqa-diamond-nemo)
  - gpqa_diamond questions, reasoning, using NeMo's alignment template
* - [gpqa_extended](#simple-evals-gpqa-extended)
  - gpqa_extended 0-shot CoT
* - [gpqa_main](#simple-evals-gpqa-main)
  - gpqa_main 0-shot CoT
* - [healthbench](#simple-evals-healthbench)
  - HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare.
* - [healthbench_consensus](#simple-evals-healthbench-consensus)
  - HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare. The consensus subset measures 34 particularly important aspects of model behavior and has been validated by the consensus of multiple physicians.
* - [healthbench_hard](#simple-evals-healthbench-hard)
  - HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare. The hard subset consists of 1000 examples chosen because they are difficult for current frontier models.
* - [humaneval](#simple-evals-humaneval)
  - HumanEval evaluates the performance in Python code generation tasks. It used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.
* - [humanevalplus](#simple-evals-humanevalplus)
  - HumanEvalPlus is a dataset of 164 programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.
* - [math_test_500](#simple-evals-math-test-500)
  - Open Ai math test 500
* - [math_test_500_nemo](#simple-evals-math-test-500-nemo)
  - math_test_500 questions, math, using NeMo's alignment template
* - [mgsm](#simple-evals-mgsm)
  - MGSM is a benchmark of grade-school math problems. The same 250 problems from GSM8K are each translated via human annotators in 10 languages.
* - [mmlu](#simple-evals-mmlu)
  - MMLU 0-shot CoT
* - [mmlu_am](#simple-evals-mmlu-am)
  - MMLU 0-shot CoT in Amharic (am)
* - [mmlu_ar](#simple-evals-mmlu-ar)
  - MMLU 0-shot CoT in Arabic (ar)
* - [mmlu_ar-lite](#simple-evals-mmlu-ar-lite)
  - Lite version of the MMLU 0-shot CoT in Arabic (ar)
* - [mmlu_bn](#simple-evals-mmlu-bn)
  - MMLU 0-shot CoT in Bengali (bn)
* - [mmlu_bn-lite](#simple-evals-mmlu-bn-lite)
  - Lite version of the MMLU 0-shot CoT in Bengali (bn)
* - [mmlu_cs](#simple-evals-mmlu-cs)
  - MMLU 0-shot CoT in Czech (cs)
* - [mmlu_de](#simple-evals-mmlu-de)
  - MMLU 0-shot CoT in German (de)
* - [mmlu_de-lite](#simple-evals-mmlu-de-lite)
  - Lite version of the MMLU 0-shot CoT in German (de)
* - [mmlu_el](#simple-evals-mmlu-el)
  - MMLU 0-shot CoT in Greek (el)
* - [mmlu_en](#simple-evals-mmlu-en)
  - MMLU 0-shot CoT in English (en)
* - [mmlu_en-lite](#simple-evals-mmlu-en-lite)
  - Lite version of the MMLU 0-shot CoT in English (en)
* - [mmlu_es](#simple-evals-mmlu-es)
  - MMLU 0-shot CoT in Spanish (es)
* - [mmlu_es-lite](#simple-evals-mmlu-es-lite)
  - Lite version of the MMLU 0-shot CoT in Spanish (es)
* - [mmlu_fa](#simple-evals-mmlu-fa)
  - MMLU 0-shot CoT in Persian (fa)
* - [mmlu_fil](#simple-evals-mmlu-fil)
  - MMLU 0-shot CoT in Filipino (fil)
* - [mmlu_fr](#simple-evals-mmlu-fr)
  - MMLU 0-shot CoT in French (fr)
* - [mmlu_fr-lite](#simple-evals-mmlu-fr-lite)
  - Lite version of the MMLU 0-shot CoT in French (fr)
* - [mmlu_ha](#simple-evals-mmlu-ha)
  - MMLU 0-shot CoT in Hausa (ha)
* - [mmlu_he](#simple-evals-mmlu-he)
  - MMLU 0-shot CoT in Hebrew (he)
* - [mmlu_hi](#simple-evals-mmlu-hi)
  - MMLU 0-shot CoT in Hindi (hi)
* - [mmlu_hi-lite](#simple-evals-mmlu-hi-lite)
  - Lite version of the MMLU 0-shot CoT in Hindi (hi)
* - [mmlu_id](#simple-evals-mmlu-id)
  - MMLU 0-shot CoT in Indonesian (id)
* - [mmlu_id-lite](#simple-evals-mmlu-id-lite)
  - Lite version of the MMLU 0-shot CoT in Indonesian (id)
* - [mmlu_ig](#simple-evals-mmlu-ig)
  - MMLU 0-shot CoT in Igbo (ig)
* - [mmlu_it](#simple-evals-mmlu-it)
  - MMLU 0-shot CoT in Italian (it)
* - [mmlu_it-lite](#simple-evals-mmlu-it-lite)
  - Lite version of the MMLU 0-shot CoT in Italian (it)
* - [mmlu_ja](#simple-evals-mmlu-ja)
  - MMLU 0-shot CoT in Japanese (ja)
* - [mmlu_ja-lite](#simple-evals-mmlu-ja-lite)
  - Lite version of the MMLU 0-shot CoT in Japanese (ja)
* - [mmlu_ko](#simple-evals-mmlu-ko)
  - MMLU 0-shot CoT in Korean (ko)
* - [mmlu_ko-lite](#simple-evals-mmlu-ko-lite)
  - Lite version of the MMLU 0-shot CoT in Korean (ko)
* - [mmlu_ky](#simple-evals-mmlu-ky)
  - MMLU 0-shot CoT in Kyrgyz (ky)
* - [mmlu_llama_4](#simple-evals-mmlu-llama-4)
  - MMLU questions with custom regex extraction patterns for Llama 4
* - [mmlu_lt](#simple-evals-mmlu-lt)
  - MMLU 0-shot CoT in Lithuanian (lt)
* - [mmlu_mg](#simple-evals-mmlu-mg)
  - MMLU 0-shot CoT in Malagasy (mg)
* - [mmlu_ms](#simple-evals-mmlu-ms)
  - MMLU 0-shot CoT in Malay (ms)
* - [mmlu_my-lite](#simple-evals-mmlu-my-lite)
  - Lite version of the MMLU 0-shot CoT in Malay (my)
* - [mmlu_ne](#simple-evals-mmlu-ne)
  - MMLU 0-shot CoT in Nepali (ne)
* - [mmlu_nl](#simple-evals-mmlu-nl)
  - MMLU 0-shot CoT in Dutch (nl)
* - [mmlu_ny](#simple-evals-mmlu-ny)
  - MMLU 0-shot CoT in Nyanja (ny)
* - [mmlu_pl](#simple-evals-mmlu-pl)
  - MMLU 0-shot CoT in Polish (pl)
* - [mmlu_pro](#simple-evals-mmlu-pro)
  - MMLU-Pro dataset is a more robust and challenging massive multi-task understanding dataset tailored to more rigorously benchmark large language models' capabilities. This dataset contains 12K complex questions across various disciplines.
* - [mmlu_pro_llama_4](#simple-evals-mmlu-pro-llama-4)
  - MMLU-Pro questions with custom regex extraction patterns for Llama 4
* - [mmlu_pt](#simple-evals-mmlu-pt)
  - MMLU 0-shot CoT in Portuguese (pt)
* - [mmlu_pt-lite](#simple-evals-mmlu-pt-lite)
  - Lite version of the MMLU 0-shot CoT in Portuguese (pt)
* - [mmlu_ro](#simple-evals-mmlu-ro)
  - MMLU 0-shot CoT in Romanian (ro)
* - [mmlu_ru](#simple-evals-mmlu-ru)
  - MMLU 0-shot CoT in Russian (ru)
* - [mmlu_si](#simple-evals-mmlu-si)
  - MMLU 0-shot CoT in Sinhala (si)
* - [mmlu_sn](#simple-evals-mmlu-sn)
  - MMLU 0-shot CoT in Shona (sn)
* - [mmlu_so](#simple-evals-mmlu-so)
  - MMLU 0-shot CoT in Somali (so)
* - [mmlu_sr](#simple-evals-mmlu-sr)
  - MMLU 0-shot CoT in Serbian (sr)
* - [mmlu_sv](#simple-evals-mmlu-sv)
  - MMLU 0-shot CoT in Swedish (sv)
* - [mmlu_sw](#simple-evals-mmlu-sw)
  - MMLU 0-shot CoT in Swahili (sw)
* - [mmlu_sw-lite](#simple-evals-mmlu-sw-lite)
  - Lite version of the MMLU 0-shot CoT in Swahili (sw)
* - [mmlu_te](#simple-evals-mmlu-te)
  - MMLU 0-shot CoT in Telugu (te)
* - [mmlu_tr](#simple-evals-mmlu-tr)
  - MMLU 0-shot CoT in Turkish (tr)
* - [mmlu_uk](#simple-evals-mmlu-uk)
  - MMLU 0-shot CoT in Ukrainian (uk)
* - [mmlu_vi](#simple-evals-mmlu-vi)
  - MMLU 0-shot CoT in Vietnamese (vi)
* - [mmlu_yo](#simple-evals-mmlu-yo)
  - MMLU 0-shot CoT in Yoruba (yo)
* - [mmlu_yo-lite](#simple-evals-mmlu-yo-lite)
  - Lite version of the MMLU 0-shot CoT in Yoruba (yo)
* - [mmlu_zh-lite](#simple-evals-mmlu-zh-lite)
  - Lite version of the MMLU 0-shot CoT in Chinese (Simplified) (zh)
* - [simpleqa](#simple-evals-simpleqa)
  - A factuality benchmark called SimpleQA that measures the ability for language models to answer short, fact-seeking questions.
```

(simple-evals-aa-aime-2024)=
## AA_AIME_2024

AA AIME 2024 questions, math

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 10
    task: AA_AIME_2024
  type: AA_AIME_2024
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-aa-math-test-500)=
## AA_math_test_500

AA Open Ai math test 500

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 3
    task: AA_math_test_500
  type: AA_math_test_500
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-aime-2024)=
## AIME_2024

AIME 2024 questions, math

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: AIME_2024
  type: AIME_2024
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-aime-2025)=
## AIME_2025

AIME 2025 questions, math

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 10
    task: AIME_2025
  type: AIME_2025
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-aime-2024-nemo)=
## aime_2024_nemo

AIME 2024 questions, math, using NeMo's alignment template

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 10
    task: aime_2024_nemo
  type: aime_2024_nemo
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-aime-2025-nemo)=
## aime_2025_nemo

AIME 2025 questions, math, using NeMo's alignment template

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 10
    task: aime_2025_nemo
  type: aime_2025_nemo
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-browsecomp)=
## browsecomp

BrowseComp is a benchmark for measuring the ability for agents to browse the web.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: browsecomp
  type: browsecomp
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-diamond)=
## gpqa_diamond

gpqa_diamond 0-shot CoT

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: gpqa_diamond
  type: gpqa_diamond
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-diamond-aa-v2)=
## gpqa_diamond_aa_v2

gpqa_diamond questions with custom regex extraction patterns for AA v2

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config:
        extraction:
        - regex: (?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])
          match_group: 1
          name: aa_v2_regex
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 5
    task: gpqa_diamond
  type: gpqa_diamond_aa_v2
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-diamond-aa-v2-llama-4)=
## gpqa_diamond_aa_v2_llama_4

gpqa_diamond questions with custom regex extraction patterns for Llama 4

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config:
        extraction:
        - regex: (?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_colon_llama4
        - regex: (?i)(?:the )?best? answer is\s*[\*\_,{}\.]*([A-D])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_is_llama4
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 5
    task: gpqa_diamond
  type: gpqa_diamond_aa_v2_llama_4
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-diamond-nemo)=
## gpqa_diamond_nemo

gpqa_diamond questions, reasoning, using NeMo's alignment template

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 5
    task: gpqa_diamond_nemo
  type: gpqa_diamond_nemo
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-extended)=
## gpqa_extended

gpqa_extended 0-shot CoT

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: gpqa_extended
  type: gpqa_extended
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-gpqa-main)=
## gpqa_main

gpqa_main 0-shot CoT

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: gpqa_main
  type: gpqa_main
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-healthbench)=
## healthbench

HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: healthbench
  type: healthbench
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-healthbench-consensus)=
## healthbench_consensus

HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare. The consensus subset measures 34 particularly important aspects of model behavior and has been validated by the consensus of multiple physicians.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: healthbench_consensus
  type: healthbench_consensus
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-healthbench-hard)=
## healthbench_hard

HealthBench is an open-source benchmark measuring the performance and safety of large language models in healthcare. The hard subset consists of 1000 examples chosen because they are difficult for current frontier models.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: JUDGE_API_KEY
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: healthbench_hard
  type: healthbench_hard
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-humaneval)=
## humaneval

HumanEval evaluates the performance in Python code generation tasks. It used to measure functional correctness for synthesizing programs from docstrings. It consists of 164 original programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: humaneval
    n_repeats: 1
  type: humaneval
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-humanevalplus)=
## humanevalplus

HumanEvalPlus is a dataset of 164 programming problems, assessing language comprehension, algorithms, and simple mathematics, with some comparable to simple software interview questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: humanevalplus
    n_repeats: 1
  type: humanevalplus
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-math-test-500)=
## math_test_500

Open Ai math test 500

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: math_test_500
  type: math_test_500
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-math-test-500-nemo)=
## math_test_500_nemo

math_test_500 questions, math, using NeMo's alignment template

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
      n_samples: 3
    task: math_test_500_nemo
  type: math_test_500_nemo
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mgsm)=
## mgsm

MGSM is a benchmark of grade-school math problems. The same 250 problems from GSM8K are each translated via human annotators in 10 languages.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mgsm
  type: mgsm
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu)=
## mmlu

MMLU 0-shot CoT

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu
  type: mmlu
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-am)=
## mmlu_am

MMLU 0-shot CoT in Amharic (am)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_am
  type: mmlu_am
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ar)=
## mmlu_ar

MMLU 0-shot CoT in Arabic (ar)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ar
  type: mmlu_ar
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ar-lite)=
## mmlu_ar-lite

Lite version of the MMLU 0-shot CoT in Arabic (ar)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ar-lite
  type: mmlu_ar-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-bn)=
## mmlu_bn

MMLU 0-shot CoT in Bengali (bn)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_bn
  type: mmlu_bn
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-bn-lite)=
## mmlu_bn-lite

Lite version of the MMLU 0-shot CoT in Bengali (bn)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_bn-lite
  type: mmlu_bn-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-cs)=
## mmlu_cs

MMLU 0-shot CoT in Czech (cs)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_cs
  type: mmlu_cs
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-de)=
## mmlu_de

MMLU 0-shot CoT in German (de)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_de
  type: mmlu_de
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-de-lite)=
## mmlu_de-lite

Lite version of the MMLU 0-shot CoT in German (de)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_de-lite
  type: mmlu_de-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-el)=
## mmlu_el

MMLU 0-shot CoT in Greek (el)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_el
  type: mmlu_el
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-en)=
## mmlu_en

MMLU 0-shot CoT in English (en)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_en
  type: mmlu_en
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-en-lite)=
## mmlu_en-lite

Lite version of the MMLU 0-shot CoT in English (en)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_en-lite
  type: mmlu_en-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-es)=
## mmlu_es

MMLU 0-shot CoT in Spanish (es)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_es
  type: mmlu_es
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-es-lite)=
## mmlu_es-lite

Lite version of the MMLU 0-shot CoT in Spanish (es)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_es-lite
  type: mmlu_es-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-fa)=
## mmlu_fa

MMLU 0-shot CoT in Persian (fa)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_fa
  type: mmlu_fa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-fil)=
## mmlu_fil

MMLU 0-shot CoT in Filipino (fil)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_fil
  type: mmlu_fil
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-fr)=
## mmlu_fr

MMLU 0-shot CoT in French (fr)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_fr
  type: mmlu_fr
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-fr-lite)=
## mmlu_fr-lite

Lite version of the MMLU 0-shot CoT in French (fr)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_fr-lite
  type: mmlu_fr-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ha)=
## mmlu_ha

MMLU 0-shot CoT in Hausa (ha)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ha
  type: mmlu_ha
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-he)=
## mmlu_he

MMLU 0-shot CoT in Hebrew (he)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_he
  type: mmlu_he
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-hi)=
## mmlu_hi

MMLU 0-shot CoT in Hindi (hi)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_hi
  type: mmlu_hi
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-hi-lite)=
## mmlu_hi-lite

Lite version of the MMLU 0-shot CoT in Hindi (hi)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_hi-lite
  type: mmlu_hi-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-id)=
## mmlu_id

MMLU 0-shot CoT in Indonesian (id)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_id
  type: mmlu_id
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-id-lite)=
## mmlu_id-lite

Lite version of the MMLU 0-shot CoT in Indonesian (id)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_id-lite
  type: mmlu_id-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ig)=
## mmlu_ig

MMLU 0-shot CoT in Igbo (ig)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ig
  type: mmlu_ig
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-it)=
## mmlu_it

MMLU 0-shot CoT in Italian (it)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_it
  type: mmlu_it
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-it-lite)=
## mmlu_it-lite

Lite version of the MMLU 0-shot CoT in Italian (it)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_it-lite
  type: mmlu_it-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ja)=
## mmlu_ja

MMLU 0-shot CoT in Japanese (ja)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ja
  type: mmlu_ja
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ja-lite)=
## mmlu_ja-lite

Lite version of the MMLU 0-shot CoT in Japanese (ja)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ja-lite
  type: mmlu_ja-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ko)=
## mmlu_ko

MMLU 0-shot CoT in Korean (ko)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ko
  type: mmlu_ko
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ko-lite)=
## mmlu_ko-lite

Lite version of the MMLU 0-shot CoT in Korean (ko)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ko-lite
  type: mmlu_ko-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ky)=
## mmlu_ky

MMLU 0-shot CoT in Kyrgyz (ky)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ky
  type: mmlu_ky
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-llama-4)=
## mmlu_llama_4

MMLU questions with custom regex extraction patterns for Llama 4

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config:
        extraction:
        - regex: (?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_colon_llama4
        - regex: (?i)(?:the )?best? answer is\s*[\*\_,{}\.]*([A-D])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_is_llama4
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu
  type: mmlu_llama_4
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-lt)=
## mmlu_lt

MMLU 0-shot CoT in Lithuanian (lt)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_lt
  type: mmlu_lt
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-mg)=
## mmlu_mg

MMLU 0-shot CoT in Malagasy (mg)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_mg
  type: mmlu_mg
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ms)=
## mmlu_ms

MMLU 0-shot CoT in Malay (ms)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ms
  type: mmlu_ms
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-my-lite)=
## mmlu_my-lite

Lite version of the MMLU 0-shot CoT in Malay (my)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_my-lite
  type: mmlu_my-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ne)=
## mmlu_ne

MMLU 0-shot CoT in Nepali (ne)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ne
  type: mmlu_ne
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-nl)=
## mmlu_nl

MMLU 0-shot CoT in Dutch (nl)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_nl
  type: mmlu_nl
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ny)=
## mmlu_ny

MMLU 0-shot CoT in Nyanja (ny)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ny
  type: mmlu_ny
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-pl)=
## mmlu_pl

MMLU 0-shot CoT in Polish (pl)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_pl
  type: mmlu_pl
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-pro)=
## mmlu_pro

MMLU-Pro dataset is a more robust and challenging massive multi-task understanding dataset tailored to more rigorously benchmark large language models' capabilities. This dataset contains 12K complex questions across various disciplines.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_pro
  type: mmlu_pro
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-pro-llama-4)=
## mmlu_pro_llama_4

MMLU-Pro questions with custom regex extraction patterns for Llama 4

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config:
        extraction:
        - regex: (?i)[\*\_]{0,2}Answer[\*\_]{0,2}\s*:[\s\*\_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_colon_llama4
        - regex: (?i)(?:the )?best? answer is\s*[\*\_,{}\.]*([A-D])(?![a-zA-Z0-9])
          match_group: 1
          name: answer_is_llama4
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_pro
  type: mmlu_pro_llama_4
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-pt)=
## mmlu_pt

MMLU 0-shot CoT in Portuguese (pt)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_pt
  type: mmlu_pt
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-pt-lite)=
## mmlu_pt-lite

Lite version of the MMLU 0-shot CoT in Portuguese (pt)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_pt-lite
  type: mmlu_pt-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ro)=
## mmlu_ro

MMLU 0-shot CoT in Romanian (ro)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ro
  type: mmlu_ro
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-ru)=
## mmlu_ru

MMLU 0-shot CoT in Russian (ru)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_ru
  type: mmlu_ru
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-si)=
## mmlu_si

MMLU 0-shot CoT in Sinhala (si)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_si
  type: mmlu_si
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-sn)=
## mmlu_sn

MMLU 0-shot CoT in Shona (sn)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_sn
  type: mmlu_sn
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-so)=
## mmlu_so

MMLU 0-shot CoT in Somali (so)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_so
  type: mmlu_so
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-sr)=
## mmlu_sr

MMLU 0-shot CoT in Serbian (sr)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_sr
  type: mmlu_sr
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-sv)=
## mmlu_sv

MMLU 0-shot CoT in Swedish (sv)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_sv
  type: mmlu_sv
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-sw)=
## mmlu_sw

MMLU 0-shot CoT in Swahili (sw)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_sw
  type: mmlu_sw
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-sw-lite)=
## mmlu_sw-lite

Lite version of the MMLU 0-shot CoT in Swahili (sw)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_sw-lite
  type: mmlu_sw-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-te)=
## mmlu_te

MMLU 0-shot CoT in Telugu (te)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_te
  type: mmlu_te
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-tr)=
## mmlu_tr

MMLU 0-shot CoT in Turkish (tr)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_tr
  type: mmlu_tr
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-uk)=
## mmlu_uk

MMLU 0-shot CoT in Ukrainian (uk)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_uk
  type: mmlu_uk
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-vi)=
## mmlu_vi

MMLU 0-shot CoT in Vietnamese (vi)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_vi
  type: mmlu_vi
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-yo)=
## mmlu_yo

MMLU 0-shot CoT in Yoruba (yo)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_yo
  type: mmlu_yo
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-yo-lite)=
## mmlu_yo-lite

Lite version of the MMLU 0-shot CoT in Yoruba (yo)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_yo-lite
  type: mmlu_yo-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-mmlu-zh-lite)=
## mmlu_zh-lite

Lite version of the MMLU 0-shot CoT in Chinese (Simplified) (zh)

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: mmlu_zh-lite
  type: mmlu_zh-lite
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>


(simple-evals-simpleqa)=
## simpleqa

A factuality benchmark called SimpleQA that measures the ability for language models to answer short, fact-seeking questions.

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-10T13-29-9db0f7ca
```

**Container Digest:**
```
sha256:bb86e9fe35452679cacd362cfcc2b9485661108569a5d57565e3275e784f7b46
```

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
config:
  params:
    limit_samples: null
    max_new_tokens: 4096
    temperature: 0
    top_p: 1.0e-05
    parallelism: 10
    max_retries: 5
    request_timeout: 60
    extra:
      downsampling_ratio: null
      add_system_prompt: false
      custom_config: null
      judge:
        url: null
        model_id: null
        api_key: null
        backend: openai
        request_timeout: 600
        max_retries: 16
        temperature: 0.0
        top_p: 0.0001
        max_tokens: 1024
        max_concurrent_requests: null
    task: simpleqa
  type: simpleqa
  supported_endpoint_types:
  - chat
target:
  api_endpoint: {}

```

</details>

