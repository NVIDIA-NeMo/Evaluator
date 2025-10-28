# Language Model Containers

Containers specialized for evaluating large language models across academic benchmarks, custom tasks, and conversation scenarios.

---

## Simple-Evals Container

**NGC Catalog**: [simple-evals](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/simple-evals)

Container for lightweight evaluation tasks and simple model assessments.

**Use Cases:**
- Simple question-answering evaluation
- Math and reasoning capabilities
- Basic Python coding

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/simple-evals:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0` |
| `top_p` | `1e-05` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `60` |
| `downsampling_ratio` | `None` |
| `add_system_prompt` | `False` |
| `custom_config` | `None` |
| `judge` | `{'url': None, 'model_id': None, 'api_key': None, 'backend': 'openai', 'request_timeout': 600, 'max_retries': 16, 'temperature': 0.0, 'top_p': 0.0001, 'max_tokens': 1024, 'max_concurrent_requests': None}` |

---

## LM-Evaluation-Harness Container

**NGC Catalog**: [lm-evaluation-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/lm-evaluation-harness)

Container based on the Language Model Evaluation Harness framework for comprehensive language model evaluation.

**Use Cases:**
- Standard NLP benchmarks
- Language model performance evaluation
- Multi-task assessment
- Academic benchmark evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/lm-evaluation-harness:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `None` |
| `temperature` | `1e-07` |
| `top_p` | `0.9999999` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `tokenizer` | `None` |
| `tokenizer_backend` | `None` |
| `downsampling_ratio` | `None` |
| `tokenized_requests` | `False` |

---

## MT-Bench Container

**NGC Catalog**: [mtbench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mtbench)

Container for MT-Bench evaluation framework, designed for multi-turn conversation evaluation.

**Use Cases:**
- Multi-turn dialogue evaluation
- Conversation quality assessment
- Context maintenance evaluation
- Interactive AI system testing

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/mtbench:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `max_new_tokens` | `1024` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `judge` | `{'url': None, 'model_id': 'gpt-4', 'api_key': None, 'request_timeout': 60, 'max_retries': 16, 'temperature': 0.0, 'top_p': 0.0001, 'max_tokens': 2048}` |

---

## HELM Container

**NGC Catalog**: [helm](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/helm)

Container for the Holistic Evaluation of Language Models (HELM) framework, with a focus on MedHELM - an extensible evaluation framework for assessing LLM performance for medical tasks.

**Use Cases:**
- Medical AI model evaluation
- Clinical task assessment
- Healthcare-specific benchmarking
- Diagnostic decision-making evaluation
- Patient communication assessment
- Medical knowledge evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/helm:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `parallelism` | `1` |
| `data_path` | `None` |
| `num_output_tokens` | `None` |
| `subject` | `None` |
| `condition` | `None` |
| `max_length` | `None` |
| `num_train_trials` | `None` |
| `subset` | `None` |
| `gpt_judge_api_key` | `GPT_JUDGE_API_KEY` |
| `llama_judge_api_key` | `LLAMA_JUDGE_API_KEY` |
| `claude_judge_api_key` | `CLAUDE_JUDGE_API_KEY` |

---

## RAG Retriever Evaluation Container

**NGC Catalog**: [rag_retriever_eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/rag_retriever_eval)

Container for evaluating Retrieval-Augmented Generation (RAG) systems and their retrieval capabilities.

**Use Cases:**
- Document retrieval accuracy
- Context relevance assessment
- RAG pipeline evaluation
- Information retrieval performance

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/rag_retriever_eval:{{ docker_compose_latest }}
```

---

## HLE Container

**NGC Catalog**: [hle](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/hle)

Container for Humanity's Last Exam (HLE), a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark with broad subject coverage.

**Use Cases:**
- Academic knowledge and problem solving evaluation
- Multi-modal benchmark testing
- Frontier knowledge assessment
- Subject-matter expertise evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/hle:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0.0` |
| `top_p` | `1.0` |
| `parallelism` | `100` |
| `max_retries` | `30` |
| `request_timeout` | `600.0` |

---

## IFBench Container

**NGC Catalog**: [ifbench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/ifbench)

Container for a challenging benchmark for precise instruction following evaluation.

**Use Cases:**
- Precise instruction following evaluation
- Out-of-distribution constraint verification
- Multiturn constraint isolation testing
- Instruction following robustness assessment
- Verifiable instruction compliance testing

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/ifbench:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0.01` |
| `top_p` | `0.95` |
| `parallelism` | `8` |
| `max_retries` | `5` |

---

## MMATH Container

**NGC Catalog**: [mmath](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/mmath)

Container for multilingual mathematical reasoning evaluation across multiple languages.

**Use Cases:**
- Multilingual mathematical reasoning evaluation
- Cross-lingual mathematical problem solving assessment
- Mathematical reasoning robustness across languages
- Complex mathematical reasoning capability testing
- Translation quality validation for mathematical content

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/mmath:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `32768` |
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `parallelism` | `8` |
| `max_retries` | `5` |
| `language` | `en` |

**Supported Languages:** EN, ZH, AR, ES, FR, JA, KO, PT, TH, VI


## ProfBench Container

**NGC Catalog**: [profbench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/profbench)

Container for assessing performance accross professional domains in business and scientific research.

**Use Cases:**
- Evaluation of professional knowledge accross Physics PhD, Chemistry PhD, Finance MBA and Consulting MBA
- Report generation capabilities
- Quality assessment of LLM judges


**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/profbench:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0.0` |
| `top_p` | `0.00001` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `600` |

---

## NeMo Skills Container

**NGC Catalog**: [nemo-skills](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/nemo-skills)

Container for assessing LLM capabilities in science, maths and agentic workflows.

**Use Cases:**
- Evaluation of reasoning capabilities
- Advanced math and coding skills
- Agentic workflow

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/nemo-skills:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `65536` |
| `temperature` | `None` |
| `top_p` | `None` |
| `parallelism` | `16` |


---