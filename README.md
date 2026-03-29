# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

## Install

```bash
pip install -e .                   # core
pip install -e ".[scoring]"        # + sympy
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
| `nel regression` | Paired regression analysis |
| `nel config` | Persistent user config |
| `nel package` | Containerize BYOB benchmark |

## Regression Analysis

Compare two evaluation runs with paired statistical testing. Detects which specific problems regressed, not just whether the overall score dropped.

### Quick start

```bash
# Terminal dashboard — verdict + capability profile + flip summary
nel regression ./baseline/eval-base.json ./candidate/eval-cand.json

# Full investigation document with per-problem tables and model responses
nel regression ./baseline/eval-base.json ./candidate/eval-cand.json \
  --show-flips --report regression_report.md

# CI/CD gate — exit 1 on BLOCK, exit 2 on WARN
nel regression ./baseline/eval-base.json ./candidate/eval-cand.json --strict

# Compact for Slack (4 lines)
nel regression ./baseline/eval-base.json ./candidate/eval-cand.json --compact

# Machine-parseable JSON to stdout
nel regression ./baseline/eval-base.json ./candidate/eval-cand.json --format json
```

### What it does

Both runs must evaluate the **same benchmark** (same `problem_idx` values). The tool:

1. **Pairs samples** by `(problem_idx, repeat)` across the two `results.jsonl` files
2. **Classifies each pair** into a 2×2 contingency table:
   - Both correct (stable) — both models got it right
   - Baseline only correct (regression) — baseline right, candidate wrong
   - Candidate only correct (improvement) — baseline wrong, candidate right
   - Both wrong (stable) — both models got it wrong
3. **Runs McNemar's exact binomial test** (one-sided, testing for degradation) on the discordant pairs
4. **Computes a verdict**: PASS / WARN / BLOCK / INCONCLUSIVE based on statistical significance AND practical effect size

### Verdicts

| Verdict | Condition | Meaning | Exit code (`--strict`) |
|---------|-----------|---------|------------------------|
| **PASS** | p ≥ α | No evidence of regression | 0 |
| **WARN** | p < α, effect ≤ ε | Significant but below practical threshold | 2 |
| **BLOCK** | p < α, effect > ε | Significant and practically meaningful | 1 |
| **INCONCLUSIVE** | Too few discordant pairs | Not enough data to detect small regressions | 2 |

Default thresholds: α = 0.05, ε = 0.05 (5% absolute drop). Override with `--max-drop`.

### Output modes

**Terminal** (default) — color-coded Capability Impact Profile with bar charts:
```
PASS — 1 regressions, 0 improvements out of 100 paired samples — not significant

CAPABILITY IMPACT PROFILE
────────────────────────────────────────────────────────────
  code_generation    ███████████████░░░░░   75.0% →  75.0%  (+0.0%)  HELD
  long_context_128k  █████████████░░░░░░░   70.0% →  65.0%  (-5.0%)  HELD
  math_reasoning     ████████████████████  100.0% → 100.0%  (+0.0%)  HELD

FLIP SUMMARY  (100 paired samples)
────────────────────────────────────────────────────────────
  Stable correct:    79  ████████████████████████  (both got it right)
  Stable wrong:      20  ░░░░░░  (both got it wrong)
  Regressions:        1  ▓  (baseline right, candidate wrong) (1.0%)
  Improvements:       0    (baseline wrong, candidate right)
```

**Markdown report** (`--report report.md`) — full investigation document with:
- Executive summary with emoji indicators
- Capability impact table
- Full McNemar statistical details and 2×2 contingency table
- Category breakdown of regressions
- Per-problem flip tables showing expected answer, model response, and category
- Investigation guidance and suggested next steps

**Compact** (`--compact`) — 4 lines for Slack or CI logs.

**JSON** (`--format json`) — structured report to stdout for downstream tooling.

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--max-drop` / `-t` | 0.05 | Maximum allowed absolute drop in any metric (0–1 scale). |
| `--correct-above` | 0.0 | Reward above this = correct. Use `0.5` for judge scores. |
| `--strict` | off | Exit 1 on BLOCK, exit 2 on WARN/INCONCLUSIVE. |
| `--show-flips` | off | Show per-sample flip list in terminal. |
| `--report PATH` | — | Write full Markdown investigation report. |
| `--compact` | off | Short output for Slack / CI (~4 lines). |
| `--format` | text | `text` or `json` (JSON to stdout). |
| `--verbose` | off | Show p-values, test method, confidence intervals. |
| `--output` / `-o` | — | Write raw JSON report to file. |

### Python API

```python
from nemo_evaluator.engine import compare_runs, compare_results, build_flip_report, mcnemar_test

# From files
report = compare_runs("baseline/eval-base.json", "candidate/eval-cand.json")
print(report["verdict"])  # "PASS" / "WARN" / "BLOCK" / "INCONCLUSIVE"

# In-memory (for library integration)
report = compare_results(baseline_records, candidate_records)

# Individual components
flip = build_flip_report(base_records, cand_records, threshold=0.5)
result = mcnemar_test(flip.contingency, flip.summary.n_paired)
```

### Statistical methodology

- **McNemar's exact binomial test** (one-sided) on paired binary outcomes, following the ICLR 2026 methodology ("When LLMs get significantly worse")
- **Clustered standard errors** per Miller 2024 ("Adding Error Bars to Evals") — accounts for within-group score correlation in benchmarks like MMLU
- **Effect size**: net regression rate `(b - c) / N` with 95% Wald CI
- **Power-aware verdicts**: INCONCLUSIVE when the test cannot detect regressions below the practical threshold

Requires `pip install nemo-evaluator[stats]` for p-values. Degrades gracefully without scipy.

## Examples

See [`examples/configs/`](examples/configs/) for 25 end-to-end configs covering all solver types, verification methods, and execution backends.

## License

Apache-2.0
