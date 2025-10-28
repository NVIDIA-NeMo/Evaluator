# Model Efficiency

Containers specialized in evaluating Large Language Model efficiency.

---

## GenAIPerf Container

**NGC Catalog**: [genai-perf](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/eval-factory/containers/genai-perf)

Container for assessing the speed of processing requests by the server.

**Use Cases:**

- Analysis time to first token (TTF) and inter-token latency (ITL)
- Assessment of server efficiency under load
- Summarization scenario: long input, short output
- Generation scenatio: short input, long output

**Pull Command:**

```bash
docker pull nvcr.io/nvidia/eval-factory/genai-perf:{{ docker_compose_latest }}
```

**Default Parameters:**

| Parameter | Value |
|-----------|-------|
| `parallelism` | `1` |

Benchmark-specific parameters (passed via `extra` field):

| Parameter | Description |
|-----------|-------|
| `tokenizer` | HuggingFace tokenizer to use for calculating the number of tokens. **Requied parameter**  (default: `None`)|
| `warmup` | Whether to run warmup (default: `True`) |
| `isl` | Input sequence length (default: task-specific, see below) |
| `osl` | Output sequence length (default: task-specific, see below) |


**Supported Benchmarks:**

- `genai_perf_summarization` - Speed analysis with `isl: 5000` and `osl: 500`.
- `genai_perf_generation` - Speed analysis with `isl: 500` and `osl: 5000`.