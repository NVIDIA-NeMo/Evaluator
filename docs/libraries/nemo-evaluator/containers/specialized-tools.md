# Specialized Tools Containers

Containers for specialized evaluation tasks including agentic AI capabilities and advanced reasoning assessments.

---

## Agentic Evaluation Container

**NGC Catalog**: [agentic_eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)

Container for evaluating agentic AI models on tool usage and planning tasks.

**Use Cases:**

- Tool usage evaluation
- Planning tasks assessment

**Pull Command:**

```bash
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:{{ docker_compose_latest }}
```

**Supported Benchmarks:**

- `agentic_eval_answer_accuracy`
- `agentic_eval_goal_accuracy_with_reference`
- `agentic_eval_goal_accuracy_without_reference`
- `agentic_eval_topic_adherence`
- `agentic_eval_tool_call_accuracy`

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
