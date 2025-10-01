# Code Generation Containers

Containers specialized for evaluating code generation models and programming language capabilities.

---

## BigCode Evaluation Harness Container

**NGC Catalog**: [bigcode-evaluation-harness](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bigcode-evaluation-harness)

Container specialized for evaluating code generation models and programming language models.

**Use Cases:**
- Code generation quality assessment
- Programming problem solving
- Code completion evaluation
- Software engineering task assessment

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `512` |
| `temperature` | `1e-07` |
| `top_p` | `0.9999999` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `30` |
| `do_sample` | `True` |
| `n_samples` | `1` |

---

## BFCL Container

**NGC Catalog**: [bfcl](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/bfcl)

Container for Berkeley Function-Calling Leaderboard evaluation framework.

**Use Cases:**
- Tool usage evaluation
- Multi-turn interactions
- Native support for function/tool calling
- Function calling evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/bfcl:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `parallelism` | `10` |
| `native_calling` | `False` |
| `custom_dataset` | `{'path': None, 'format': None, 'data_template_path': None}` |

---

## ToolTalk Container

**NGC Catalog**: [tooltalk](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/tooltalk)

Container for evaluating AI models' ability to use tools and APIs effectively.

**Use Cases:**
- Tool usage evaluation
- API interaction assessment
- Function calling evaluation
- External tool integration testing

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/tooltalk:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |

---

## LiveCodeBench Container

**NGC Catalog**: [livecodebench](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/livecodebench)

LiveCodeBench provides holistic and contamination-free evaluation of coding capabilities of LLMs. It continuously collects new problems from contests across three competition platforms -- LeetCode, AtCoder, and CodeForces.

**Use Cases:**
- Holistic coding capability evaluation
- Contamination-free assessment
- Contest-style problem solving
- Code generation and execution
- Test output prediction
- Self-repair capabilities

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/livecodebench:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `4096` |
| `temperature` | `0.0` |
| `top_p` | `1e-05` |
| `parallelism` | `10` |
| `max_retries` | `5` |
| `request_timeout` | `60` |
| `n_samples` | `10` |
| `num_process_evaluate` | `5` |
| `cache_batch_size` | `10` |
| `support_system_role` | `False` |
| `cot_code_execution` | `False` |

**Supported Versions:** v1-v6, 0724_0125, 0824_0225

---

## SciCode Container

**NGC Catalog**: [scicode](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/scicode)

SciCode is a challenging benchmark designed to evaluate the capabilities of language models in generating code for solving realistic scientific research problems with diverse coverage across 16 subdomains from 6 domains.

**Use Cases:**
- Scientific research code generation
- Multi-domain scientific programming
- Research workflow automation
- Scientific computation evaluation
- Domain-specific coding tasks

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/scicode:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `temperature` | `0` |
| `max_new_tokens` | `2048` |
| `top_p` | `1e-05` |
| `request_timeout` | `60` |
| `max_retries` | `2` |
| `with_background` | `False` |
| `include_dev` | `False` |
| `n_samples` | `1` |
| `eval_threads` | `None` |

**Supported Domains:** Physics, Math, Material Science, Biology, Chemistry (16 subdomains from 5 domains)
