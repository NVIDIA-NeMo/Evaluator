# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

## Install

```bash
pip install -e .                   # core
pip install -e ".[scoring]"        # + sympy
pip install -e ".[scoring,stats]"  # + sympy + scipy
pip install -e ".[harbor]"         # + Harbor agents
pip install -e ".[all]"            # everything
```

## Quick Start

```bash
nel eval run --bench mmlu \
  --model-url https://api.example.com/v1 \
  --model-id my-model \
  --repeats 3 --max-problems 100

nel eval run config.yaml
nel eval run config.yaml --resume
nel eval report ./eval_results/ -f markdown -o report.md
```

## Benchmarks

15 built-in benchmarks plus external harness integrations:

| Benchmark | Type | Scoring |
|-----------|------|---------|
| mmlu, mmlu_pro, gpqa | Multichoice | `multichoice_regex` |
| gsm8k, math500, mgsm | Math | `numeric_match` / `answer_line` |
| drop, triviaqa | QA | `fuzzy_match` |
| humaneval | Code | `code_sandbox` (Docker) |
| simpleqa, healthbench | Judge | `needs_judge` |
| pinchbench | Agentic | `code_sandbox` / `needs_judge` |
| xstest | Safety | `needs_judge` |
| swebench-verified, swebench-multilingual | SWE | Docker two-container |

External environments via URI schemes: `lm-eval://`, `skills://`, `vlmevalkit://`, `gym://`, `harbor://`, `container://`.

## BYOB

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, exact_match

@benchmark(name="my-bench", dataset="hf://my-org/data?split=test",
           prompt="Q: {question}\nA:", target_field="answer")
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return exact_match(sample)
```

## Solvers

| Solver | Endpoint Type | Use Case |
|--------|---------------|----------|
| `ChatSolver` | `chat` | Standard chat completions (default) |
| `CompletionSolver` | `completions` | Base model text completions |
| `VLMSolver` | `vlm` | Vision-language (images + text) |
| `HarborSolver` | `harbor` | Harbor agents (OpenHands etc.) |
| `GymSolver` | `gym` | Delegate to nemo-gym server |
| `NatSolver` | `nat` | NeMo Agent Toolkit (SSE) |
| `OpenClawSolver` | `openclaw` | OpenClaw CLI agent |

### Compatibility

| Environment | chat | completions | vlm | harbor | gym | nat | openclaw |
|-------------|------|-------------|-----|--------|-----|-----|----------|
| skills://      | yes  | yes  | yes  | yes    | yes | yes | yes      |
| lm-eval://     | yes  | yes  | yes  | yes    | yes | yes | yes      |
| vlmevalkit://  | --   | --   | yes  | --     | --  | --  | --       |
| gym://         | yes  | yes  | yes  | yes    | yes | yes | yes      |
| BYOB           | yes  | yes  | yes  | yes    | yes | yes | yes      |
| harbor://      | yes* | yes* | yes* | yes    | yes | yes | yes      |
| swebench-*     | yes* | yes* | yes* | yes    | yes | yes | yes      |
| container://   | --   | --   | --   | --     | --  | --  | --       |
| pinchbench     | partial | partial | partial | yes | yes | yes | yes |

`yes*` = two-container mode; `partial` = only LLM-as-judge tasks score meaningfully; `--` = incompatible.

## Sandboxes

Per-problem Docker/SLURM sandboxes for code execution and agentic evaluation. Two modes: **stateful** (shared sandbox for solve+verify) and **stateless** (separate agent and verification containers with shared volume).

## SLURM

Pyxis/Enroot-based execution with auto-selected container images per URI scheme. Supports colocated and separated (`nel serve`) environment modes.

| Tag suffix | Contents |
|------------|----------|
| `:latest` | Base + gym + vlmevalkit |
| `:latest-lm-eval` | + lm-evaluation-harness |
| `:latest-skills` | + NeMo Skills |
| `:latest-full` | All harnesses |

## CLI

| Command | Purpose |
|---------|---------|
| `nel eval run` | Run evaluation (name or YAML) |
| `nel eval report <dir>` | Generate reports |
| `nel list` | List benchmarks |
| `nel serve -b <name>` | Serve as HTTP endpoint |
| `nel validate -b <name>` | Sanity check |
| `nel regression` | Compare runs (p-values) |
| `nel config` | Persistent user config |
| `nel package` | Containerize BYOB benchmark |

## Examples

See [`examples/configs/`](examples/configs/) for 16 end-to-end configs covering all solver types, verification methods, and execution backends.

## License

Apache-2.0
