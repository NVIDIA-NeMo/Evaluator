# Harness Integration Report: simple-evals & lm-evaluation-harness

## Problem Statement

The current approach in `nemo_evaluator/benchmarks/` copy-pastes benchmark logic: each benchmark (GSM8K, MMLU-Pro, MATH, HumanEval, IFEval) reimplements dataset loading, prompt formatting, and scoring. This is:

- **Fragile**: Our copy diverges from upstream the moment they fix a bug or update a dataset.
- **Incomplete**: simple-evals has 10 evals, lm-eval-harness has 600+. We have 6. Reimplementing each one is not scalable.
- **Redundant**: The scoring logic we vendored (`mcq.py`, `math_equal.py`) duplicates what these repos already do correctly.

The RFC says: *"Adapters for external environments are a transitional necessity -- thin protocol translation to consume environments that already exist."* We should take that seriously.

## Design Principle

**We are NOT an eval harness. We are the evaluation authority that sits on top of eval harnesses.**

Instead of reimplementing benchmarks, we inject ourselves into the model call path of existing harnesses. This gives us:

1. Full request/response trajectories (we **are** the model from the harness's perspective)
2. Per-sample scoring details (the harness scores; we capture)
3. Zero maintenance of benchmark code (upstream owns it)
4. Instant access to all upstream benchmarks (10 from simple-evals, 600+ from lm-eval)

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                  NeMo Evaluator (NEL)                 │
│                                                      │
│  nel run --harness simple-evals --eval mmlu           │
│  nel run --harness lm-eval --tasks arc_easy,mmlu      │
│                                                      │
│  ┌────────────────────────────────────────────────┐  │
│  │            Harness Adapter Layer                │  │
│  │                                                │  │
│  │  SimpleEvalsAdapter    LMEvalHarnessAdapter     │  │
│  │  - Implements           - Implements            │  │
│  │    SamplerBase            LM (generate_until,   │  │
│  │  - Routes chat()          loglikelihood)         │  │
│  │    through ModelClient  - Routes through        │  │
│  │  - Captures per-call      ModelClient           │  │
│  │    StepRecords          - Captures per-call     │  │
│  │                           StepRecords           │  │
│  └───────────────┬────────────────────────────────┘  │
│                  │                                    │
│  ┌───────────────▼────────────────────────────────┐  │
│  │         ModelClient (our model proxy)           │  │
│  │  - Caching, retries, connection pooling         │  │
│  │  - Full request/response logging                │  │
│  │  - Token counting, latency tracking             │  │
│  └───────────────┬────────────────────────────────┘  │
│                  │                                    │
│  ┌───────────────▼────────────────────────────────┐  │
│  │      Decision-Grade Layer                       │  │
│  │  - Confidence intervals over harness scores     │  │
│  │  - pass@k from n-repeats                        │  │
│  │  - Per-category breakdowns                      │  │
│  │  - Regression detection                         │  │
│  │  - Artifact bundle                              │  │
│  └────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
            │
            ▼
  ┌──────────────────────┐
  │  Real Model Endpoint │
  │  (vLLM, NIM, API)    │
  └──────────────────────┘
```

---

## Integration 1: simple-evals

### How it works today

```python
# simple-evals pattern
eval_obj = MathEval(num_examples=500)
result = eval_obj(sampler)  # sampler: SamplerBase → SamplerResponse
```

- `SamplerBase.__call__(messages: list[dict]) -> SamplerResponse`
- Each eval loads its own dataset, formats prompts, calls the sampler, scores.
- `EvalResult` has per-sample scores, HTML, conversation logs, and aggregate metrics.

### Our adapter: `NelSampler(SamplerBase)`

We implement `SamplerBase` and route through our `ModelClient`. The harness calls us for every model interaction; we log everything.

```python
from simple_evals.types import SamplerBase, SamplerResponse, MessageList

class NelSampler(SamplerBase):
    """Bridge: simple-evals calls this; we route through ModelClient and capture."""
    
    def __init__(self, client: ModelClient):
        self.client = client
        self.call_log: list[StepRecord] = []
    
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        # Convert simple-evals messages to our format, call model, capture response
        resp = asyncio.run(self.client.chat(messages=message_list))
        # Log full trajectory
        self.call_log.append(StepRecord(
            prompt=message_list[-1]["content"],
            model_response=resp,
            ...
        ))
        return SamplerResponse(
            response_text=resp.content,
            actual_queried_message_list=message_list,
            response_metadata={"tokens": resp.total_tokens, "latency_ms": resp.latency_ms},
        )
```

### Discovery: listing available evals

```python
def list_simple_evals() -> list[dict]:
    """Discover available evals from simple-evals repo without importing each one."""
    import importlib, inspect
    from simple_evals.types import Eval
    
    evals = []
    for name in ["math_eval", "mmlu_eval", "gpqa_eval", "drop_eval", 
                  "mgsm_eval", "humaneval_eval", "simpleqa_eval", "browsecomp_eval"]:
        mod = importlib.import_module(f"simple_evals.{name}")
        for attr_name, cls in inspect.getmembers(mod, inspect.isclass):
            if issubclass(cls, Eval) and cls is not Eval:
                evals.append({"name": name.replace("_eval", ""), "class": cls, "module": name})
    return evals
```

### Running: the full loop

```python
def run_simple_eval(eval_name: str, client: ModelClient, n_repeats: int = 1, 
                    num_examples: int | None = None) -> dict:
    # 1. Discover and instantiate
    eval_cls = get_simple_eval(eval_name)
    eval_obj = eval_cls(num_examples=num_examples) if num_examples else eval_cls()
    
    all_step_records = []
    all_scores = []
    
    for rep in range(n_repeats):
        # 2. Create our sampler bridge
        sampler = NelSampler(client)
        
        # 3. Run the eval (simple-evals owns the loop, scoring, everything)
        result: EvalResult = eval_obj(sampler)
        
        # 4. Capture results
        all_step_records.extend(sampler.call_log)
        all_scores.append(result.score)
    
    # 5. Apply decision-grade layer on top of harness scores
    ci = bootstrap_ci(all_scores)
    return build_artifact_bundle(eval_name, all_step_records, ...)
```

### What we get

| Artifact | Source |
|----------|--------|
| Per-sample scores | `EvalResult.htmls`, per-`SingleEvalResult.score` |
| Full trajectories | Our `NelSampler.call_log` (every model call captured) |
| Token counts / latency | `ModelResponse` from our `ModelClient` |
| Scoring details | `SingleEvalResult.metrics` (e.g. `{"equality_check": True}`) |
| Category breakdowns | From `SingleEvalResult.example_level_metadata` |
| CIs / pass@k | Our decision-grade layer on top of raw scores |
| Caching | Our `ResponseCache` (the sampler goes through `ModelClient`) |

### Zero changes to simple-evals

The simple-evals repo is used as a **pip dependency** (or git submodule). We never modify it. We only implement `SamplerBase`.

---

## Integration 2: lm-evaluation-harness

### How it works today

```python
import lm_eval

results = lm_eval.simple_evaluate(
    model="openai-chat-completions",  # or custom LM subclass
    model_args="model=gpt-4",
    tasks=["arc_easy", "mmlu"],
    log_samples=True,
)
# results["results"]["arc_easy"]["acc"] = 0.85
```

- `LM` abstract base class requires: `generate_until()`, `loglikelihood()`, `loglikelihood_rolling()`
- Tasks are YAML-defined; 600+ available via `TaskManager.all_tasks`
- `EvalResults` has per-task metrics, per-sample data when `log_samples=True`

### Our adapter: `NelLM(LM)`

```python
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance

class NelLM(LM):
    """Bridge: lm-eval calls this; we route through ModelClient and capture."""
    
    def __init__(self, client: ModelClient):
        super().__init__()
        self.client = client
        self.call_log: list[StepRecord] = []
    
    def generate_until(self, requests: list[Instance]) -> list[str]:
        results = []
        for req in requests:
            context, gen_kwargs = req.args
            resp = asyncio.run(self.client.chat(prompt=context))
            self.call_log.append(...)
            results.append(resp.content)
        return results
    
    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        # For logprob-based tasks, we need a completion API
        # Fallback: use generate_until and return (0.0, False) for unsupported
        # Or: route to a vLLM/local endpoint that provides logprobs
        ...
    
    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        ...
```

### Discovery: listing available tasks

```python
def list_lm_eval_tasks() -> list[str]:
    from lm_eval.tasks import TaskManager
    tm = TaskManager()
    return tm.all_tasks  # 600+ tasks
```

### Running: the full loop

```python
def run_lm_eval(tasks: list[str], client: ModelClient, n_repeats: int = 1,
                num_fewshot: int | None = None, limit: int | None = None) -> dict:
    import lm_eval
    
    all_results = []
    for rep in range(n_repeats):
        nel_lm = NelLM(client)
        
        results = lm_eval.simple_evaluate(
            model=nel_lm,           # Pass our LM instance directly
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            log_samples=True,
        )
        all_results.append(results)
    
    # Decision-grade layer over n-repeats
    return build_multi_task_bundle(all_results, nel_lm.call_log)
```

### Limitations: loglikelihood tasks

Many lm-eval tasks use `loglikelihood` (measuring token probabilities), which requires access to model logits -- not available through a chat API. Two approaches:

1. **Filter to `generate_until` tasks only** for chat API models. These are the tasks that matter most (MMLU, ARC, GSM8K, BBH, etc. all have `generate_until` variants).
2. **Support `loglikelihood` when the model provides it** (vLLM, local HF models). The `NelLM` can be extended with a `completions` endpoint.

This is the same limitation that lm-eval itself has with API models.

---

## Integration 3: What about our existing benchmarks?

The copy-pasted benchmarks in `nemo_evaluator/benchmarks/` should be **deprecated in favor of harness adapters**. However, they serve two purposes the adapters don't:

1. **Gym-compatible `nel serve`**: Our benchmarks expose `seed()`/`verify()` endpoints that Gym can consume. The harness adapters don't decompose into seed/verify -- they run the whole eval.
2. **BYOB reference implementations**: They show how to write an `EvalEnvironment` for a new benchmark.

**Recommendation**: Keep a small set of native BYOB benchmarks (AIME, a few custom ones) as reference implementations and for `nel serve`. For everything else, use the harness adapters.

---

## CLI Design

```bash
# simple-evals
nel run --harness simple-evals --eval mmlu --examples 500
nel run --harness simple-evals --eval math,gpqa,humaneval --repeats 4
nel harness list --harness simple-evals

# lm-evaluation-harness
nel run --harness lm-eval --tasks arc_easy,mmlu,gsm8k_cot --fewshot 5
nel run --harness lm-eval --tasks leaderboard --limit 1000 --repeats 4
nel harness list --harness lm-eval

# Mixed: run simple-evals math + lm-eval arc in one report
nel run config.yaml  # where config references both harnesses
nel report ./eval_results/  # unified multi-harness report
```

Config YAML:

```yaml
evaluation:
  tasks:
    - harness: simple-evals
      eval: math
      examples: 500
      repeats: 4
    - harness: simple-evals
      eval: mmlu
      repeats: 4
    - harness: lm-eval
      tasks: [arc_easy, arc_challenge, hellaswag]
      fewshot: 5
      repeats: 1
    - harness: lm-eval
      tasks: [gsm8k_cot]
      fewshot: 8
      repeats: 4
    - type: environment
      benchmark: AIME_2025  # native BYOB
      repeats: 8
```

---

## Dependency Strategy

Both repos are **optional dependencies**:

```toml
[project.optional-dependencies]
simple-evals = ["simple-evals @ git+https://github.com/openai/simple-evals.git"]
lm-eval = ["lm_eval>=0.4"]
harnesses = ["nemo-evaluator[simple-evals,lm-eval]"]
```

The adapters do lazy imports: if you `nel run --harness lm-eval` without `lm_eval` installed, you get a clear error telling you to `pip install nemo-evaluator[lm-eval]`.

---

## Comparison: Copy-Paste vs. Harness Adapter

| Dimension | Copy-Paste (current) | Harness Adapter (proposed) |
|-----------|---------------------|---------------------------|
| Benchmarks available | 2 BYOB (GSM8K, TriviaQA) | 10 (simple-evals) + 600+ (lm-eval) |
| Scoring correctness | Our reimplementation (risk of bugs) | Upstream's battle-tested scoring |
| Maintenance burden | We own all scoring logic | Zero -- upstream maintains |
| Upstream sync | Manual, error-prone | Automatic via version pin |
| Few-shot support | We had to build it | lm-eval has it built-in |
| Prompt templates | We reimplemented | Upstream Jinja2 templates |
| Dataset loading | We reimplemented | Upstream handles it |
| Full observability | Yes (we own the loop) | Yes (we are the model) |
| n-repeats | Yes | Yes (we wrap) |
| CIs / pass@k | Yes | Yes (our layer on top) |
| Gym serve | Yes (`seed`/`verify`) | No (these are run-only adapters) |
| Latency overhead | ~0 | ~0 (same model calls) |

---

## What we keep in `nemo_evaluator/scoring/`

The scoring subpackage still has value for:

1. **BYOB benchmarks** that are natively authored with `EvalEnvironment` (e.g., custom internal benchmarks, AIME, customer-data evals).
2. **Cross-validation**: RFC Section 7 says we can cross-check a harness's reward against our own scorer. `math_equal` and `mcq_score` are useful for this.
3. **Gym `nel serve` environments**: These need `seed()`/`verify()` decomposition.

But `scoring/` should not be the mechanism for consuming standard benchmarks. That's what the harness adapters are for.

---

## Implementation Plan

### Phase 1: simple-evals adapter (1-2 days)

1. `NelSampler(SamplerBase)` -- routes through `ModelClient`, captures `StepRecord`s
2. `SimpleEvalsAdapter` -- discovers evals, runs them, wraps results into bundles
3. `nel run --harness simple-evals --eval <name>` CLI support
4. `nel harness list --harness simple-evals`
5. n-repeats + decision-grade layer on top

### Phase 2: lm-eval adapter (2-3 days)

1. `NelLM(LM)` -- implements `generate_until`, routes through `ModelClient`
2. `LMEvalAdapter` -- wraps `simple_evaluate()`, captures results
3. `nel run --harness lm-eval --tasks <names>` CLI support
4. `nel harness list --harness lm-eval`
5. Handle `loglikelihood` gracefully (skip or support via completions endpoint)

### Phase 3: Unified config + report (1 day)

1. Config YAML supports `harness:` field alongside existing `benchmark:` and `adapter:`
2. `nel report` aggregates results across harness types
3. Documentation and examples

### Phase 4: Clean up copy-pasted benchmarks (DONE)

1. Removed `mmlu_pro.py`, `math_bench.py`, `humaneval.py`, `ifeval.py`
2. Kept `gsm8k.py` and `triviaqa.py` as BYOB reference templates
3. Standard benchmarks are available through harness adapters

---

## Summary

**Stop reimplementing benchmarks. Start wrapping harnesses.**

The harness adapter pattern gives us:
- 610+ benchmarks overnight (vs. 6 today)
- Zero scoring maintenance
- Full observability (we are the model proxy)
- Decision-grade layer on top of battle-tested scoring
- Clean separation: harnesses own benchmarks, we own evaluation authority
