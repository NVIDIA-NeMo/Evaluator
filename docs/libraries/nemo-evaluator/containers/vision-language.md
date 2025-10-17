# Vision-Language Containers

Containers specialized for evaluating multimodal models that process both visual and textual information.

---

## VLMEvalKit Container

**NGC Catalog**: [vlmevalkit](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/vlmevalkit)

Container for Vision-Language Model evaluation toolkit.

**Use Cases:**

- Multimodal model evaluation
- Image-text understanding assessment
- Visual reasoning evaluation
- Cross-modal performance testing

**Pull Command:**

```bash
docker pull nvcr.io/nvidia/eval-factory/vlmevalkit:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `limit_samples` | `None` |
| `max_new_tokens` | `2048` |
| `temperature` | `0` |
| `top_p` | `None` |
| `parallelism` | `4` |
| `max_retries` | `5` |
| `request_timeout` | `60` |

**Supported Benchmarks:**

- `ocrbench` - Optical character recognition and text understanding
- `slidevqa` - Slide-based visual question answering (requires `OPENAI_CLIENT_ID`, `OPENAI_CLIENT_SECRET`)
- `chartqa` - Chart and graph interpretation
- `ai2d_judge` - AI2 Diagram understanding (requires `OPENAI_CLIENT_ID`, `OPENAI_CLIENT_SECRET`)
