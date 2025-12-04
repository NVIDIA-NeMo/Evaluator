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
* - [AIME_2025_aa_v2](#simple-evals-aime-2025-aa-v2)
  - AIME 2025 questions, math - params aligned with Artificial Analysis Index v2
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
* - [mgsm_aa_v2](#simple-evals-mgsm-aa-v2)
  - MGSM is a benchmark of grade-school math problems - params aligned with Artificial Analysis Index v2
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
* - [mmlu_pro_aa_v2](#simple-evals-mmlu-pro-aa-v2)
  - MMLU-Pro - params aligned with Artificial Analysis Index v2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `AA_AIME_2024`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: AA_AIME_2024
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: AA_AIME_2024
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `AA_math_test_500`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: AA_math_test_500
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: AA_math_test_500
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `AIME_2024`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: AIME_2024
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: AIME_2024
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `AIME_2025`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: AIME_2025
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: AIME_2025
target:
  api_endpoint: {}
```

</details>


(simple-evals-aime-2025-aa-v2)=
## AIME_2025_aa_v2

AIME 2025 questions, math - params aligned with Artificial Analysis Index v2

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `AIME_2025_aa_v2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 16384
    max_retries: 30
    parallelism: 10
    task: AIME_2025
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: AIME_2025_aa_v2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `aime_2024_nemo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: aime_2024_nemo
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: aime_2024_nemo
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `aime_2025_nemo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: aime_2025_nemo
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: aime_2025_nemo
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `browsecomp`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: browsecomp
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: browsecomp
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_diamond`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: gpqa_diamond
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_diamond
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_diamond_aa_v2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 16384
    max_retries: 30
    parallelism: 10
    task: gpqa_diamond
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_diamond_aa_v2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_diamond_aa_v2_llama_4`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: gpqa_diamond
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_diamond_aa_v2_llama_4
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_diamond_nemo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: gpqa_diamond_nemo
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_diamond_nemo
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_extended`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: gpqa_extended
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_extended
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `gpqa_main`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: gpqa_main
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: gpqa_main
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `healthbench`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: healthbench
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: healthbench
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `healthbench_consensus`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: healthbench_consensus
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: healthbench_consensus
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `healthbench_hard`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: healthbench_hard
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: healthbench_hard
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `humaneval`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: humaneval
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
      n_samples: 1
  supported_endpoint_types:
  - chat
  type: humaneval
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `humanevalplus`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: humanevalplus
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
      n_samples: 1
  supported_endpoint_types:
  - chat
  type: humanevalplus
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `math_test_500`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: math_test_500
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: math_test_500
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `math_test_500_nemo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: math_test_500_nemo
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: math_test_500_nemo
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mgsm`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mgsm
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mgsm
target:
  api_endpoint: {}
```

</details>


(simple-evals-mgsm-aa-v2)=
## mgsm_aa_v2

MGSM is a benchmark of grade-school math problems - params aligned with Artificial Analysis Index v2

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mgsm_aa_v2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 16384
    max_retries: 30
    parallelism: 10
    task: mgsm
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mgsm_aa_v2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_am`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_am
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_am
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ar`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ar
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ar
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ar-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ar-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ar-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_bn`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_bn
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_bn
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_bn-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_bn-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_bn-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_cs`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_cs
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_cs
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_de`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_de
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_de
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_de-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_de-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_de-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_el`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_el
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_el
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_en`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_en
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_en
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_en-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_en-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_en-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_es`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_es
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_es
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_es-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_es-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_es-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_fa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_fa
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_fa
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_fil`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_fil
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_fil
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_fr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_fr
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_fr
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_fr-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_fr-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_fr-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ha`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ha
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ha
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_he`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_he
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_he
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_hi`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_hi
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_hi
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_hi-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_hi-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_hi-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_id`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_id
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_id
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_id-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_id-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_id-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ig`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ig
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ig
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_it`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_it
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_it
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_it-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_it-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_it-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ja`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ja
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ja
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ja-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ja-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ja-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ko`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ko
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ko
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ko-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ko-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ko-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ky`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ky
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ky
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_llama_4`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_llama_4
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_lt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_lt
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_lt
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_mg`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_mg
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_mg
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ms`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ms
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ms
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_my-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_my-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_my-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ne`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ne
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ne
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_nl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_nl
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_nl
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ny`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ny
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ny
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pl`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_pl
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pl
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pro`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_pro
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pro
target:
  api_endpoint: {}
```

</details>


(simple-evals-mmlu-pro-aa-v2)=
## mmlu_pro_aa_v2

MMLU-Pro - params aligned with Artificial Analysis Index v2

<details>
<summary><strong>View task details</strong></summary>

**Harness:** `simple-evals`

**Container:**
```
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pro_aa_v2`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 16384
    max_retries: 30
    parallelism: 10
    task: mmlu_pro
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pro_aa_v2
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pro_llama_4`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_pro
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pro_llama_4
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pt`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_pt
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pt
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_pt-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_pt-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_pt-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ro`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ro
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ro
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_ru`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_ru
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_ru
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_si`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_si
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_si
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_sn`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_sn
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_sn
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_so`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_so
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_so
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_sr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_sr
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_sr
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_sv`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_sv
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_sv
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_sw`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_sw
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_sw
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_sw-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_sw-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_sw-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_te`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_te
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_te
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_tr`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_tr
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_tr
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_uk`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_uk
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_uk
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_vi`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_vi
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_vi
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_yo`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_yo
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_yo
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_yo-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_yo-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_yo-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `mmlu_zh-lite`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: mmlu_zh-lite
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: mmlu_zh-lite
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
gitlab-master.nvidia.com:5005/dl/joc/competitive_evaluation/nvidia-core-evals/ci-llm/simple-evals:dev-2025-11-27T12-26-df507571
```

**Container Digest:**
```
sha256:45a3d86af3dee4e2ab5aef3823f6e338c6817f9f927605341ad51423a3757ae8
```

**Task Type:** `simpleqa`

**Command:**
```bash
{% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} python3 -c 'import yaml, json, sys; config_data = {{config.params.extra.custom_config | tojson}}; json.dump(config_data, open("{{config.output_dir}}/temp_config.json", "w")); yaml.dump(config_data, open("{{config.output_dir}}/custom_config.yml", "w"), default_flow_style=False)' && {% endif %} simple_evals --model {{target.api_endpoint.model_id}} --eval_name {{config.params.task}} --url {{target.api_endpoint.url}} --temperature {{config.params.temperature}} --top_p {{config.params.top_p}} --max_tokens {{config.params.max_new_tokens}} --out_dir {{config.output_dir}}/{{config.type}} --cache_dir {{config.output_dir}}/{{config.type}}/cache --num_threads {{config.params.parallelism}} --max_retries {{config.params.max_retries}} --timeout {{config.params.request_timeout}} {% if config.params.extra.n_samples is defined %} --num_repeats {{config.params.extra.n_samples}}{% endif %} {% if config.params.limit_samples is not none %} --first_n {{config.params.limit_samples}}{% endif %} {% if config.params.extra.add_system_prompt  %} --add_system_prompt {% endif %} {% if config.params.extra.downsampling_ratio is not none %} --downsampling_ratio {{config.params.extra.downsampling_ratio}}{% endif %} {% if config.params.extra.args is defined %} {{ config.params.extra.args }} {% endif %} {% if config.params.extra.judge.url is not none %} --judge_url {{config.params.extra.judge.url}}{% endif %} {% if config.params.extra.judge.model_id is not none %} --judge_model_id {{config.params.extra.judge.model_id}}{% endif %} {% if config.params.extra.judge.api_key is not none %} --judge_api_key_name {{config.params.extra.judge.api_key}}{% endif %} {% if config.params.extra.judge.backend is not none %} --judge_backend {{config.params.extra.judge.backend}}{% endif %} {% if config.params.extra.judge.request_timeout is not none %} --judge_request_timeout {{config.params.extra.judge.request_timeout}}{% endif %} {% if config.params.extra.judge.max_retries is not none %} --judge_max_retries {{config.params.extra.judge.max_retries}}{% endif %} {% if config.params.extra.judge.temperature is not none %} --judge_temperature {{config.params.extra.judge.temperature}}{% endif %} {% if config.params.extra.judge.top_p is not none %} --judge_top_p {{config.params.extra.judge.top_p}}{% endif %} {% if config.params.extra.judge.max_tokens is not none %} --judge_max_tokens {{config.params.extra.judge.max_tokens}}{% endif %} {% if config.params.extra.judge.max_concurrent_requests is not none %} --judge_max_concurrent_requests {{config.params.extra.judge.max_concurrent_requests}}{% endif %} {% if config.params.extra.custom_config is defined and config.params.extra.custom_config is not none %} --custom_eval_cfg_file {{config.output_dir}}/custom_config.yml{% endif %}
```

**Defaults:**
```yaml
framework_name: simple_evals
pkg_name: simple_evals
config:
  params:
    max_new_tokens: 4096
    max_retries: 5
    parallelism: 10
    task: simpleqa
    temperature: 0.0
    request_timeout: 60
    top_p: 1.0e-05
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
  supported_endpoint_types:
  - chat
  type: simpleqa
target:
  api_endpoint: {}
```

</details>

