# Specialized Tools Containers

Containers for specialized evaluation tasks including agentic AI capabilities and advanced reasoning assessments.

---

## Agentic Evaluation Container

**NGC Catalog**: [agentic_eval](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/agentic_eval)

Container for evaluating agentic AI models that can perform complex, multi-step tasks and demonstrate reasoning capabilities.

**Use Cases:**
- Multi-step problem solving
- Tool usage evaluation
- Reasoning chain assessment
- Agent behavior analysis

**Pull Command:**
```bash
docker pull nvcr.io/nvidia/eval-factory/agentic_eval:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `parallelism` | `null` |
| `judge_model_type` | `openai` |
| `judge_model_args` | `null` |
| `judge_sanity_check` | `True` |
| `metric_mode` | `"f1"` |
| `data_template_path` | `null` |

**Key Features:**
- Multi-step reasoning evaluation
- Tool interaction assessment
- Agent planning capabilities
- Complex task orchestration
- Reasoning chain analysis

**Evaluation Capabilities:**
- **Planning and Execution**: Assess how well agents can break down complex tasks
- **Tool Usage**: Evaluate ability to select and use appropriate tools
- **Reasoning Chains**: Analyze multi-step reasoning processes
- **Error Recovery**: Test agent ability to recover from failures
- **Goal Achievement**: Measure success in completing complex objectives

**Supported Agent Types:**
- Reasoning Agents
- Tool-Using Agents
- Planning Agents
- Multi-Modal Agents
- Conversational Agents
