# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

## Install

```bash
pip install -e .                # core
pip install -e ".[scoring]"     # + sympy
pip install -e ".[all]"         # everything
```

## Quick Start

```bash
# Evaluate a model on MMLU
nel eval run --bench mmlu \
  --model-url https://api.example.com/v1 \
  --model-id my-model \
  --repeats 3 --max-problems 100

# Multiple benchmarks from YAML config
nel eval run config.yaml

# Generate report
nel eval report ./eval_results/ -f markdown -o report.md
```

## Available Benchmarks

| Benchmark | Command | Type |
|-----------|---------|------|
| MMLU | `nel eval run --bench mmlu` | Multichoice |
| MMLU-Pro | `nel eval run --bench mmlu_pro` | Multichoice (10 choices) |
| MATH-500 | `nel eval run --bench math500` | Math |
| GPQA Diamond | `nel eval run --bench gpqa` | Multichoice (shuffled) |
| GSM8K | `nel eval run --bench gsm8k` | Math |
| DROP | `nel eval run --bench drop` | Reading comprehension |
| MGSM | `nel eval run --bench mgsm` | Multilingual math |
| TriviaQA | `nel eval run --bench triviaqa` | Factual QA |
| HumanEval | `nel eval run --bench humaneval` | Code (Docker sandbox) |
| SimpleQA | `nel eval run --bench simpleqa` | Factuality (judge) |
| HealthBench | `nel eval run --bench healthbench` | Health (judge) |

Plus: any lm-eval task (`nel eval run --bench lm-eval/aime25`), NeMo Skills benchmark (`nel eval run --bench skills://mmlu-pro`), or remote Gym environment (`nel eval run --bench gym://host:port`).

## Write a Benchmark in 5 Minutes

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, exact_match

@benchmark(
    name="my-bench",
    dataset="hf://my-org/my-dataset?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

That's it. Run with `nel eval run --bench my-bench`.

### Scoring Primitives

| Function | Use case |
|----------|----------|
| `exact_match(sample)` | Normalized string equality |
| `multichoice_regex(sample)` | Extract A-D/A-J from "Answer: X" |
| `answer_line(sample)` | Extract answer after "Answer:" line |
| `numeric_match(sample)` | Last number in response |
| `fuzzy_match(sample)` | Substring containment |
| `code_sandbox(sample)` | Docker-sandboxed code execution |
| `needs_judge(sample)` | Flag for LLM-as-judge post-processing |

### Extension Hooks

For benchmarks that need custom dataset preparation or prompt construction:

```python
@benchmark(
    name="gpqa",
    dataset="hf://Idavidrein/gpqa?config=gpqa_diamond&split=train",
    prompt=PROMPT_TEMPLATE,
    target_field="answer",
    prepare_row=shuffle_choices,   # transform each row after loading
    seed_fn=custom_seed,           # fully custom seed (overrides prompt template)
)
```

## Solvers

The eval loop is decoupled from inference. Plug in any solver:

| Solver | Use case |
|--------|----------|
| `ChatSolver` | Single `/chat/completions` call (default) |
| `CompletionSolver` | `/completions` endpoint |
| `AgentSolver` | External agent (OpenHands, SWE-agent, etc.) |

## Executors

| Executor | What it does |
|----------|-------------|
| `local` | In-process eval, optional Docker model deployment (default) |
| `docker` | Model server + eval in Docker containers |
| `slurm` | SLURM sbatch with model server + eval jobs |

```bash
nel eval run --bench mmlu --executor slurm --deploy nim --deploy-image nvcr.io/nim/llama3-8b
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `nel eval run --bench <name>` | Run evaluation |
| `nel eval run config.yaml` | Run from YAML config |
| `nel eval status -o <dir>` | Check running evaluation |
| `nel eval stop -o <dir>` | Stop evaluation |
| `nel eval report <dir>` | Generate reports |
| `nel list` | List available benchmarks |
| `nel serve -b <name>` | Serve benchmark as HTTP endpoint |
| `nel validate -b <name>` | Quick sanity check |
| `nel regression` | Compare two runs |

## Project Structure

```
src/nemo_evaluator/
    environments/
        base.py           EvalEnvironment base class
        define.py         @benchmark + @scorer decorator API
        registry.py       Resolution: names, URIs, namespaces
        gym.py            Gym + ManagedGym environments
        skills.py         NeMo Skills environments
        lm_eval.py        lm-evaluation-harness tasks
        pi.py             Prime Intellect environments
        server.py         HTTP server (Gym protocol)
    benchmarks/           11 built-in benchmarks (all @benchmark + @scorer)
    scoring/
        types.py          ScorerInput dataclass
        text.py           exact_match, fuzzy_match
        pattern.py        multichoice_regex, answer_line, numeric_match
        sandbox.py        code_sandbox (Docker execution)
        judge.py          LLM-as-judge (needs_judge, judge_score)
        json_schema.py    JSON schema validation
    runner/
        solver.py         Solver protocol (Chat, Completion, Agent)
        eval_loop.py      Async parallel eval with back-pressure
        model_client.py   HTTP client with retry, cache, connection pool
        deployment.py     Model server lifecycle (NIM, vLLM, Docker)
    executors/            Local, Docker, SLURM orchestration
    observability/        StepRecord, RuntimeStats, FailureReport
    metrics/              pass@k, bootstrap CI, aggregation
    cli/                  nel eval, serve, validate, list, regression
```

## License

Apache-2.0
