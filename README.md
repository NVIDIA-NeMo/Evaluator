# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

## Install

```bash
pip install -e .                   # core
pip install -e ".[scoring]"        # + sympy
pip install -e ".[stats]"          # + scipy (regression analysis)
pip install -e ".[scoring,stats]"  # + sympy + scipy
pip install -e ".[harbor]"         # + Harbor agents
pip install -e ".[proxy]"          # + LiteLLM proxy
pip install -e ".[inspect]"        # + Inspect AI log export
pip install -e ".[all]"            # everything
```

## Quick Start

```bash
nel eval run --bench mmlu \
  --model-url https://integrate.api.nvidia.com/v1 \
  --model-id nvidia/nemotron-3-super-120b-a12b \
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

## LiteLLM Proxy

Optional local proxy for LLM traffic observability. Intercepts all agent-to-model requests for logging, debugging, and custom transformations.

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}
    interceptors:
      - name: my_module.MyCallback
    proxy_verbose: true   # stream proxy logs to stderr
```

Install with `pip install -e ".[proxy]"`. Works with both local and ECS Fargate sandboxes (reverse SSH tunnel established automatically).

## Export

Evaluation results can be exported to experiment trackers and compatible formats:

```yaml
output:
  export: [inspect, wandb, mlflow]
```

- **`inspect`** -- Produces `inspect_ai`-compatible `EvalLog` JSON files. Install with `pip install -e ".[inspect]"`.
- **`wandb`** / **`mlflow`** -- Push scores and artifacts to experiment trackers. Install with `pip install -e ".[export]"`.

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

Configured via `solver.type` in each benchmark:

| Solver Type | Config `type` | Use Case |
|-------------|---------------|----------|
| SimpleSolver | `simple` | Standard chat/completion/VLM (default) |
| HarborSolver | `harbor` | Harbor agents (OpenHands, Terminus-2, etc.) |
| ToolCallingSolver | `tool_calling` | Tool-use with Gym resource servers |
| GymDelegationSolver | `gym_delegation` | Delegate to nemo-gym server |
| OpenClawSolver | `openclaw` | OpenClaw CLI agent |
| ContainerSolver | `container` | Legacy container harness |

### Compatibility

| Environment | simple | harbor | tool_calling | gym_delegation | openclaw |
|-------------|--------|--------|-------------|----------------|----------|
| BYOB           | yes | yes | yes | yes | yes |
| skills://      | yes | yes | yes | yes | yes |
| lm-eval://     | yes | yes | yes | yes | yes |
| vlmevalkit://  | yes | --  | --  | --  | --  |
| gym://         | yes | yes | yes | yes | yes |
| swebench-*     | yes* | yes | yes | yes | yes |
| container://   | --  | --  | --  | --  | --  |

`yes*` = two-container mode; `--` = incompatible.

## Sandboxes

Per-problem Docker/SLURM sandboxes for code execution and agentic evaluation. Two modes: **stateful** (shared sandbox for solve+verify) and **stateless** (separate agent and verification containers with shared volume).

## SLURM

Pyxis/Enroot-based execution with auto-selected container images per URI scheme. Uses `node_pools` topology for flexible resource allocation across model, agent, and sandbox nodes.

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
| `nel eval merge <dir>` | Merge sharded results |
| `nel eval report <dir>` | Generate reports |
| `nel list` | List benchmarks |
| `nel serve -b <name>` | Serve as HTTP endpoint |
| `nel validate -b <name>` | Sanity check |
| `nel compare` | Paired run comparison |
| `nel gate` | Multi-benchmark quality gate |
| `nel config` | Persistent user config |
| `nel package` | Containerize BYOB benchmark |

## Compare Results Between Runs

Use `nel compare` when you want to compare two runs of the same benchmark and inspect score deltas, flips, and statistical evidence.

```bash
nel compare ./results/baseline ./results/candidate --strict
```

Full tutorial: [`docs/tutorials/compare.md`](docs/tutorials/compare.md)

## Implement Quality Gates

Use `nel gate` when you want one `GO / NO-GO / INCONCLUSIVE` decision across multiple benchmarks from an explicit policy file.

```bash
nel gate ./results/baseline ./results/candidate \
  --policy gate_policy.yaml \
  --strict \
  --output gate_report.json
```

Full tutorial: [`docs/tutorials/quality-gate.md`](docs/tutorials/quality-gate.md)

## Examples

See [`examples/configs/`](examples/configs/) for 25 end-to-end configs covering all solver types, verification methods, and execution backends.

## License

Apache-2.0
