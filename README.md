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
nel run --env mmlu \
  --model-url https://api.example.com/v1 \
  --model-id my-model \
  --repeats 3 --max-problems 100

# Multiple benchmarks from YAML config
nel run config.yaml

# Generate report
nel report ./eval_results/ -f markdown -o report.md
```

## Available Benchmarks

| Benchmark | Command | Type |
|-----------|---------|------|
| MMLU | `nel run --env mmlu` | Multichoice |
| MMLU-Pro | `nel run --env mmlu_pro` | Multichoice (10 choices) |
| MATH-500 | `nel run --env math500` | Math |
| GPQA Diamond | `nel run --env gpqa` | Multichoice (shuffled) |
| GSM8K | `nel run --env gsm8k` | Math |
| DROP | `nel run --env drop` | Reading comprehension |
| MGSM | `nel run --env mgsm` | Multilingual math |
| TriviaQA | `nel run --env triviaqa` | Factual QA |
| HumanEval | `nel run --env humaneval` | Code (Docker sandbox) |
| SimpleQA | `nel run --env simpleqa` | Factuality (judge) |
| HealthBench | `nel run --env healthbench` | Health (judge) |

Plus: any lm-eval task (`nel run --env lm-eval/aime25`), NeMo Skills benchmark (`nel run --env skills://mmlu-pro`), or remote Gym environment (`nel run --env gym://host:port`).

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

That's it. Run with `nel run --env my-bench`.

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
nel run --env mmlu --executor slurm --deploy nim --deploy-image nvcr.io/nim/llama3-8b
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `nel run --env <name>` | Run evaluation |
| `nel run config.yaml` | Run from YAML config |
| `nel serve --benchmark <name>` | Serve as HTTP environment (Gym-compatible) |
| `nel validate -b <name>` | Quick sanity check (5 samples) |
| `nel report <dirs> -f markdown` | Generate comparison report |
| `nel regression baseline.json candidate.json` | Detect regressions |
| `nel harness list` | List lm-eval tasks |

## Project Structure

```
src/nemo_evaluator/
    environments/
        base.py           EvalEnvironment base class
        definitions.py    @benchmark + @scorer API
        registry.py       Resolution: names, URIs, namespaces
        gym.py            Gym + ManagedGym environments
        skills.py         NeMo Skills environments
        lm_eval.py        lm-evaluation-harness tasks
        pi.py             Prime Intellect environments
        server.py         HTTP server (Gym protocol)
    benchmarks/           11 built-in benchmarks (all @benchmark + @scorer)
    runner/
        solver.py         Solver protocol (Chat, Completion, Agent)
        eval_loop.py      Async parallel eval with back-pressure
        model_client.py   HTTP client with retry, cache, connection pool
        deployment.py     Model server lifecycle (NIM, vLLM, Docker)
    executors/            Local, Docker, SLURM orchestration
    scoring/              Judge + JSON schema scoring
    observability/        StepRecord, RuntimeStats, FailureReport
    metrics/              pass@k, bootstrap CI, aggregation
    cli/                  nel run, serve, validate, report, regression
```

## License

Apache-2.0
