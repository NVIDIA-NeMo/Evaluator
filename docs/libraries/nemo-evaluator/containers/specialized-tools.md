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
