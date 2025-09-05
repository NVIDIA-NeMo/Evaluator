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
docker pull nvcr.io/nvidia/eval-factory/bigcode-evaluation-harness:25.07.3
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

**NGC Catalog**: [bfcl](https://catalog.ngc.nvidia.com/teams/eval-factory/containers/bfcl)

Container for Berkeley Function-Calling Leaderboard evaluation framework.

**Use Cases:**
- Tool usage evaluation
- Multi-turn interactions
- Native support for function/tool calling
- Function calling evaluation

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/bfcl:25.07.3
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
docker pull nvcr.io/nvidia/eval-factory/tooltalk:25.07.1
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
