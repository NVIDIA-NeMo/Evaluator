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

**Key Features:**
- Support for various vision-language benchmarks
- Image preprocessing and encoding
- Text-image alignment evaluation
- Visual question answering capabilities
- Cross-modal retrieval assessment

**Supported Evaluation Types:**
- Visual Question Answering (VQA)
- Image Captioning
- Visual Reasoning
- Cross-modal Retrieval
- Multimodal Understanding Tasks
