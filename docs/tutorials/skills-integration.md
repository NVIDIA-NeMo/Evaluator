# NeMo Skills Integration

Use NeMo Skills benchmarks as `SkillsEnvironment` instances with full per-request observability.

## Setup

```bash
pip install -e ".[skills]"  # or: pip install nemo-skills
ns prepare_data gpqa
ns prepare_data aime24
```

## CLI

```bash
nel eval run --bench skills://gpqa --repeats 4 --max-problems 100
nel eval run --bench skills://aime24 --repeats 8
nel eval run --bench skills://mmlu --repeats 1
```

The `skills://` URI scheme resolves to a `SkillsEnvironment`, which:

1. Loads the dataset via `nemo_skills.dataset.utils`
2. Auto-prepares missing datasets
3. Selects the correct scoring method from the benchmark's `METRICS_TYPE`

## How It Works

```{mermaid}
flowchart LR
    NEL["nel eval run --bench skills://gpqa"] --> REG["Registry"]
    REG --> SE["SkillsEnvironment"]
    SE --> DS["nemo_skills dataset"]
    SE --> EVAL["nemo_skills evaluator"]
    NEL --> SOLVER["Solver"]
    SOLVER --> MC["ModelClient"]
    NEL --> OBS["Full trajectories<br/>latency, tokens, failures"]
```

`SkillsEnvironment` is a standard `EvalEnvironment`. The eval loop calls `seed(idx)` and `verify(response, expected)` like any other environment. NEL owns the model call, so you get full observability.

```{mermaid}
sequenceDiagram
    participant E as eval_loop
    participant S as SkillsEnvironment
    participant NS as nemo_skills
    participant M as Model

    E->>S: seed(idx)
    S->>NS: dataset[idx]
    S-->>E: SeedResult(prompt, expected)

    E->>M: solver.solve(task)
    M-->>E: SolveResult

    E->>S: verify(response, expected)
    S->>NS: evaluator.eval_single()
    S-->>E: VerifyResult(reward, details)
```

## Scoring

The environment automatically selects the correct scoring method based on `METRICS_TYPE`:

| Metrics type | Scoring | Benchmarks |
|-------------|---------|------------|
| `math` | Symbolic comparison via `math_equal` (sympy) | GSM8K, AIME, MATH, HMMT |
| `multichoice` | Letter extraction + exact match | GPQA, MMLU, MMLU-Pro, ARC |
| `simpleqa` | Exact match | SimpleQA, TriviaQA |
| `code_metrics` | Code execution sandbox | LiveCodeBench, EvalPlus |
| `answer-judgement` | LLM judge | Arena, AlpacaEval |

## Custom Prompt Templates

Override the default prompt for any Skills benchmark:

```bash
nel eval run --bench skills://gpqa --system-prompt "Think step by step."
```

Or via Python:

```python
from nemo_evaluator.environments.skills import SkillsEnvironment

env = SkillsEnvironment(
    "gpqa",
    prompt_template="Answer the following question.\n\n{problem}\n\nAnswer:",
)
```

## Available Benchmarks

```bash
nel list  # shows built-in + skills if installed
```

### Math Reasoning

| Benchmark | Skills name | Type | Problems |
|-----------|------------|------|----------|
| AIME 2024 | `aime24` | math | ~30 |
| AIME 2025 | `aime25` | math | ~30 |
| HMMT Feb 2025 | `hmmt_feb2025` | math | ~30 |
| GSM8K | `gsm8k` | math | 1,319 |
| MATH | `math` | math | 5,000 |

### Knowledge and Reasoning

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

## Distributed Evaluation

Combine with sharding for large benchmarks:

Use SLURM sharding via a config file (see {doc}`distributed-eval`), or use environment variables directly:


```bash
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=8 nel eval run --bench skills://mmlu --repeats 4
```

## Python API

```python
import asyncio
from nemo_evaluator.environments.skills import SkillsEnvironment
from nemo_evaluator import run_evaluation, ChatSolver, ModelClient

env = SkillsEnvironment("gpqa")
client = ModelClient(base_url="https://api.example.com/v1", model="my-model")
solver = ChatSolver(client)

bundle = asyncio.run(run_evaluation(env, solver, n_repeats=4, max_problems=100))
```
