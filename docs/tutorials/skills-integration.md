# NeMo Skills Integration

Use all 85+ NeMo Skills benchmarks with full per-request observability.

## Two Integration Modes

```{mermaid}
flowchart TB
    subgraph "Native Mode (recommended)"
        SA["SkillsAdapter<br/>skills://gpqa"] --> DS["nemo_skills dataset"]
        SA --> EVAL["nemo_skills evaluator"]
        NEL["NEL eval loop"] --> SA
        NEL --> MC["ModelClient"]
        NEL --> OBS["Full trajectories<br/>latency, tokens, failures"]
    end

    subgraph "Container Mode (legacy)"
        CA["nel container-eval<br/>nemo_skills.ns_gpqa"] --> CONT["nemo-skills container"]
        CONT --> AGG["Aggregate scores only"]
    end

    style SA fill:#e8f5e9
    style OBS fill:#e8f5e9
    style CA fill:#fff3e0
```

| Mode | Observability | Flexibility | Setup |
|------|--------------|-------------|-------|
| **Native** (`skills://`) | Full: trajectories, CI, failures, n_repeats | Full: any prompt, any model | `pip install nemo-skills` |
| **Container** | Aggregate scores + response stats | Limited: harness-controlled | Docker + NVCR access |

## Native Mode: Full Observability

### Setup

```bash
pip install nemo-skills
ns prepare_data gpqa    # download dataset
ns prepare_data aime24
ns prepare_data mmlu
```

### CLI

```bash
# Single benchmark
nel run --adapter skills://gpqa --repeats 4 --max-problems 100

# Math benchmark with custom prompt
nel run --adapter skills://aime24 --repeats 8
```

### Python API

```python
import asyncio
from nemo_evaluator.adapters.skills import SkillsAdapter
from nemo_evaluator.runner import run_evaluation, ModelClient, write_all

adapter = SkillsAdapter("gpqa")
client = ModelClient(
    base_url="https://inference-api.nvidia.com/v1",
    model="azure/openai/gpt-5.2",
)

bundle = asyncio.run(run_evaluation(
    adapter, client,
    n_repeats=4,
    max_problems=100,
))

write_all(bundle, "./results/gpqa")
```

### Available benchmarks

```bash
nel list-skills
```

Example output:

```
Available Skills benchmarks:
  aime24                         type=math            splits=[test]
  aime25                         type=math            splits=[test]
  gpqa                           type=multichoice     splits=[diamond, main]
  gsm8k                          type=math            splits=[test]
  hmmt_feb2025                   type=math            splits=[test]
  livecodebench                  type=code_metrics    splits=[test]
  mmlu                           type=multichoice     splits=[test]
  mmlu_pro                       type=multichoice     splits=[test]
```

### How scoring works

The adapter automatically selects the correct scoring method based on the benchmark's `METRICS_TYPE`:

```{mermaid}
flowchart LR
    BM["Benchmark config<br/>METRICS_TYPE"] --> |math| MATH["math_equal()<br/>sympy comparison"]
    BM --> |multichoice| MC["MCQ extraction<br/>letter matching"]
    BM --> |code_metrics| CODE["Code evaluator"]
    BM --> |other| CUSTOM["Skills evaluator<br/>eval_single()"]
```

| Metrics type | Scoring | Benchmarks |
|-------------|---------|------------|
| `math` | Symbolic comparison via math_equal (sympy) | GSM8K, AIME, MATH, HMMT |
| `multichoice` | Letter extraction + exact match | GPQA, MMLU, MMLU-Pro, ARC |
| `simpleqa` | Exact match | SimpleQA, TriviaQA |
| `code_metrics` | Code execution sandbox | LiveCodeBench, EvalPlus |
| `answer-judgement` | LLM judge | Arena, AlpacaEval |

### Custom prompt templates

Override the default prompt for any benchmark:

```python
adapter = SkillsAdapter(
    "gpqa",
    prompt_template="Answer the following question. Think step by step.\n\n{problem}\n\nAnswer:",
)
```

## Distributed Skills Evaluation

Combine with sharding for large benchmarks:

```bash
# SLURM: 16-way sharded MMLU evaluation
nel slurm eval --benchmark skills://mmlu \
    --shards 16 --repeats 4 \
    --partition batch --submit
```

Or via env vars:

```bash
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=8 nel run --adapter skills://mmlu --repeats 4
```

## Container Mode: Quick Aggregate Scores

When you need to run an evaluation exactly as the legacy evaluator runs it (same prompt templates, same scoring, same harness logic):

```bash
nel container-eval nemo_skills.ns_gpqa \
    --model-url https://inference-api.nvidia.com/v1 \
    --model-id azure/openai/gpt-5.2 \
    --limit-samples 50
```

See {doc}`legacy-containers` for details.

## Skills Benchmark Catalog

### Math Reasoning

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| AIME 2024 | `aime24` | math | ~30 |
| AIME 2025 | `aime25` | math | ~30 |
| HMMT Feb 2025 | `hmmt_feb2025` | math | ~30 |
| GSM8K | `gsm8k` | math | 1,319 |
| MATH | `math` | math | 5,000 |

### Knowledge & Reasoning

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| GPQA Diamond | `gpqa` | multichoice | ~198 |
| MMLU | `mmlu` | multichoice | ~14,000 |
| MMLU-Pro | `mmlu_pro` | multichoice | ~12,000 |
| HLE | `hle` | varies | ~500 |
| SimpleQA | `simpleqa` | simpleqa | ~4,000 |

### Code

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| LiveCodeBench | `livecodebench` | code_metrics | varies |
| SciCode | `scicode` | code_metrics | varies |
| SWE-bench | `swe_bench` | code_metrics | varies |

### Instruction Following

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| IFBench | `ifbench` | varies | varies |
| IFEval | `ifeval` | varies | varies |

### Tool Calling

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| BFCL v3 | `bfcl_v3` | varies | varies |
| BFCL v4 | `bfcl_v4` | varies | varies |

### Long Context

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| RULER | `ruler` | varies | varies |
| AA-LCR | `aa_lcr` | varies | varies |
