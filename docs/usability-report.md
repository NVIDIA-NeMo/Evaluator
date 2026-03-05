# Usability Report: NeMo Evaluator – Persona Roleplaying Sessions

**Date:** 2026-02-25
**Method:** Walkthrough of the actual `nemo-evaluator-next` codebase as each RFC persona, executing real commands and inspecting real code paths.

---

## Persona A: Research / Training Owner

> *"I own post-training. I need decision-grade scores, regression detection, and checkpoint selection. I don't want to learn a new framework -- give me one command."*

### Session: "Evaluate my model on GSM8K"

**What I tried:**
```bash
nel run --benchmark gsm8k --repeats 4 --model-url https://inference-api.nvidia.com/v1/chat/completions --model-id azure/openai/gpt-5.2
```

**What happened:** 404 error. `ModelClient` appends `/chat/completions` to `base_url`, so the URL becomes `.../v1/chat/completions/chat/completions`.

**Severity:** CRITICAL. The first thing every user does is point at a model, and the most natural URL to paste is the full endpoint.

**Fix:** Either strip `/chat/completions` from the URL automatically, or document clearly that `--model-url` expects the base (e.g. `https://inference-api.nvidia.com/v1`).

---

**What I tried next:** Fix the URL, re-run with 4 repeats on 50 problems.

**What worked well:**
- Progress bar shows live tokens/s, accuracy, ETA -- this is exactly what I need during a long run
- pass@1 and pass@4 with confidence intervals in the output -- this is the "decision-grade" quality the RFC promises
- `runtime_stats.json` shows latency percentiles and token distributions -- useful for cost estimation
- `failure_analysis.json` categorizes model errors (timeouts, refusals, empty responses) -- I can tell ops about this

**Gap 1: Multi-task config only runs first task.**
I wrote a config with GSM8K + TriviaQA (RFC Story 1 shows multi-environment YAML). Only GSM8K ran. The RFC promises unified multi-environment evaluation in one command. This is a significant gap for VPR model cards that need scores across 5+ benchmarks.

**Gap 2: No `nel regression` CLI.**
RFC Story 2 shows `--compare /results/baseline/`. The only way to compare runs today is calling `compare_runs()` in Python or via the GitLab CI template. I shouldn't need to write code for the most common post-eval action.

**Gap 3: No HTML report or model card output.**
RFC promises `output: formats: [html, yaml, json]` and `model_card: true`. Today I get JSON + JSONL. For VPR, I need something I can hand to legal. Not blocking, but a noticeable gap.

**Gap 4: No periodic evaluation during training (Story 4).**
`trigger: every_n_steps` is RFC-only. No callback mechanism exists. This is complex and probably Phase 2, but the RFC positions it as a core Persona A story.

### Persona A Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| First-run experience | **Poor** | Model URL double-path is a showstopper |
| Single benchmark eval | **Good** | Works well once URL is right; rich output |
| Multi-benchmark eval | **Not implemented** | Only first task in config runs |
| Regression detection | **Partial** | Library exists, no CLI exposure |
| Artifact quality | **Good** | Trajectories, runtime stats, failure analysis all work |
| VPR/model card readiness | **Not implemented** | No HTML/YAML report generation |

---

## Persona B: Environment / Task Developer

> *"I build reward functions and environments. I iterate fast. I need to validate my environment works before anyone uses it in training."*

### Session: "Write a new benchmark"

**What I tried:** Followed the BYOB template in `deployment-patterns.md`. Created a 30-line `EvalEnvironment` subclass.

**What worked well:**
- `@register("my_benchmark")` + `nel validate --benchmark my_benchmark --samples 10` is genuinely a 2-minute loop
- `nel validate` catches schema issues, verifies some rewards are non-zero -- exactly what I need before scaling
- `seed()` / `verify()` contract is simple and obvious
- `nel serve --benchmark my_benchmark` instantly gives me an HTTP endpoint

**Gap 1: TriviaQA has a scoring bug.**
`seed()` stores `aliases` in the row but doesn't pass `_aliases` in `SeedResult.metadata`. `verify()` looks for `metadata["_aliases"]` and falls back to `[expected]` only. This means multi-alias matching never triggers. This is the kind of bug that erodes trust -- if a built-in benchmark has silent scoring errors, how can I trust the framework for my own environments?

**Gap 2: No environment testing framework.**
RFC Story 5 shows `nel validate` catching "schema errors, timeout failures, and scoring anomalies." The current `validate` command runs problems and reports accuracy, but doesn't check things like: are all reward values in [0,1]? Do categories have consistent names? Does `seed(idx)` return deterministic results? A `--strict` mode or test harness for environment authors would save significant debugging time.

**Gap 3: `nel serve` for Gym-compat needs manual testing.**
There's no `nel test-serve` that spins up the server and fires a few requests to verify the HTTP contract. I have to manually curl or write a test script. The `validate` command only tests the Python API, not the served HTTP API.

**Gap 4: `examples/run_gsm8k.py` is broken.**
It imports `write_artifact_bundle` and `write_results_jsonl` which don't exist in `runner/artifacts.py`. If I copy this example as a starting point for my own evaluation script, it crashes immediately.

### Persona B Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| BYOB authoring experience | **Excellent** | 30 lines, register, validate, done |
| Validate command | **Good** | Works, but could be stricter |
| Built-in benchmark quality | **Has bugs** | TriviaQA alias scoring broken |
| Example scripts | **Broken** | run_gsm8k.py ImportError |
| Environment-to-serve workflow | **Good** | One command, but no HTTP contract test |

---

## Persona C: Evaluation Owner / Benchmark Engineer

> *"I curate the official benchmark suite. I need versioned graders, governed access, and reproducible scores across teams."*

### Session: "Publish a suite and run it for VPR"

**What I tried:** Wrote a `suite.yaml` with multiple benchmarks per RFC Story 7.

**Blockers:**
- No `nel publish` command exists
- No suite concept in the codebase -- only individual benchmarks
- No grader versioning (`math_equal` has no version, no calibration evidence)
- `nel list-environments` shows individual benchmarks, not suites

**What does work:**
- Can run individual benchmarks reproducibly (config hash in artifact bundle ensures traceability)
- `regression.py` provides statistical comparison with CI overlap
- Artifact bundles include config snapshots for reproducibility

**Gap 1: No suite / catalog concept.**
The RFC describes `nel publish --suite nemotron-release-v3` with versioned graders and calibration. None of this exists. For a Benchmark Engineer, the entire governance layer is missing: no suite definitions, no grader versioning, no access control, no cross-team reproducibility guarantees beyond "run the same command."

**Gap 2: No agentic evaluation (Story 8).**
SWE-Bench with sandboxes, multi-turn agents, `environment_deps` -- none of this is implemented. This is explicitly Phase 3 in the RFC, so expected, but a Benchmark Engineer looking at the tool today would see zero agent eval capability.

**Gap 3: No adapter for Harbor.**
RFC lists Harbor as a protocol (`--protocol harbor`). The codebase has Gym and PI adapters but no Harbor adapter. If I own the evaluation suite and need to include Harbor datasets, I can't.

**Gap 4: Scoring primitives are minimal.**
Only `math_equal`, `exact_match`, and basic extraction. No MCQ extraction, no LLM-as-judge, no code execution scoring. For a suite that includes MMLU, coding benchmarks, and instruction-following tasks, I'd need to write all of these myself.

### Persona C Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| Suite management | **Not implemented** | No publish, no catalog, no versioning |
| Grader governance | **Not implemented** | No versioning, no calibration |
| Benchmark execution | **Works** | Individual benchmarks run correctly |
| Cross-team reproducibility | **Partial** | Config hash helps, but no suite pinning |
| Agentic evaluation | **Not implemented** | Expected (Phase 3) |
| Protocol coverage | **Partial** | Gym + PI, no Harbor |

---

## Persona D: Platform / SRE Engineer (implicit)

> *"I deploy this on our infrastructure. I need containers, health checks, and it shouldn't surprise me at 3am."*

### Session: "Deploy on K8s for the training team"

**What worked well:**
- `Dockerfile` is clean, minimal, works
- K8s manifests have readiness/liveness probes on `/health`
- `serve-deployment.yaml` has proper resource requests/limits
- SLURM scripts handle conda activation and logging
- Shard env vars (`NEL_SHARD_IDX`/`NEL_TOTAL_SHARDS`) are a clean abstraction across all orchestrators
- Docker Compose profiles for different modes (serve, eval, sharded)

**Gap 1: No container registry / pre-built image.**
I have to build the image myself. No CI pipeline publishes to a registry. For a platform team, `docker pull nvcr.io/nemo-evaluator:v0.1` is the expected experience.

**Gap 2: No Prometheus / metrics endpoint.**
`/health` exists, but there's no `/metrics` for Prometheus scraping. For a long-running `nel serve`, I want to see request rates, latency histograms, error counts. The observability layer collects this data internally but doesn't expose it over HTTP.

**Gap 3: `nel serve` has no graceful shutdown.**
No SIGTERM handler. In K8s with `terminationGracePeriodSeconds`, in-flight `/verify` requests could be dropped.

**Gap 4: No resource estimation guidance.**
The K8s manifests request 2 CPU / 8Gi for eval. Is that enough for a 14k-problem benchmark? What about memory for TriviaQA's full HuggingFace download? No guidance on sizing.

**Gap 5: Ray not in dependencies.**
`deploy/ray_eval.py` imports `ray` but `pyproject.toml` doesn't list it. The deploy script would fail on a clean install.

### Persona D Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| Container story | **Good** | Dockerfile works, manifests are reasonable |
| Health checks | **Good** | `/health` endpoint exists |
| Observability for ops | **Missing** | No Prometheus metrics, no structured logging |
| Dependency hygiene | **Has issues** | Ray, scipy not in deps |
| Production hardening | **Partial** | No graceful shutdown, no resource guidance |

---

## Persona E: Gym Training Engineer (implicit)

> *"I run RL training on Gym. I need eval environments that plug into my training loop without modifying my code."*

### Session: "Use Evaluator benchmarks in Gym training"

**What I tried:**
```bash
nel serve --benchmark gsm8k --gym-compat --port 9090
```

**What worked well:**
- `--gym-compat` flag makes it speak Gym's `BaseVerifyRequest` protocol
- `/seed_session`, `/verify`, `/health`, `/dataset_size` all present
- `GymHarness.export_jsonl()` generates input for `ng_collect_rollouts`
- The serve endpoint works with the standard Gym config YAML pointing at it

**Gap 1: No automatic Gym resource server registration.**
I have to manually add the endpoint URL to my Gym config. The RFC envisions Evaluator benchmarks appearing in Gym's catalog automatically. Today it's manual configuration.

**Gap 2: Dual-use scoring not exposed.**
RFC Section 4.3 describes "training rewards (flexible) vs. decision-grade scores (rigid)." The serve endpoint returns a single `reward` field. There's no mechanism to configure which scoring mode to use, or to return both training and eval rewards simultaneously.

**Gap 3: No streaming / batch verify.**
Gym's `ng_collect_rollouts` sends batches. The current `/verify` endpoint handles one request at a time. For high-throughput training, this becomes a bottleneck on the Evaluator side. No batch endpoint exists.

### Persona E Verdict

| Aspect | Rating | Notes |
|--------|--------|-------|
| Gym protocol compatibility | **Good** | `--gym-compat` works |
| Integration friction | **Low** | One command to serve, manual config in Gym |
| Scoring flexibility | **Not implemented** | No dual training/eval reward |
| Performance at scale | **Untested** | No batch endpoint, unknown throughput |

---

# Consolidated Findings

## Critical Bugs -- ALL FIXED

| # | Issue | Fix |
|---|-------|-----|
| 1 | ~~Model URL double-path~~ | `ModelClient` auto-strips `/chat/completions`; all configs updated to use base URL |
| 2 | ~~`examples/run_gsm8k.py` broken imports~~ | Uses `write_all()` now |
| 3 | ~~TriviaQA aliases never passed~~ | `seed()` now passes `_aliases` in metadata |
| 4 | ~~`benchmarks` as top-level package~~ | Moved to `nemo_evaluator.benchmarks` subpackage |
| 5 | ~~Multi-task config only runs first task~~ | `nel run` now iterates all tasks in config |
| 6 | ~~`nel regression` CLI missing~~ | Added `nel regression BASELINE CANDIDATE [--strict]` |
| 7 | ~~Ray/scipy not in deps~~ | Added as optional: `pip install nemo-evaluator[ray]`, `[stats]`, or `[all]` |
| 8 | ~~PIAdapter not exported~~ | Lazy-loaded in `adapters/__init__.py` |
| 9 | ~~`sys.path` hacks in examples~~ | Removed; examples use proper `import nemo_evaluator.*` |

## Remaining Gaps

| # | Gap | Persona | RFC Story |
|---|-----|---------|-----------|
| 1 | Harbor adapter | C | Story 1 |
| 2 | HTML report / model card output | A | Story 2 |

## Medium-Priority Gaps

| # | Gap | Persona | Notes |
|---|-----|---------|-------|
| 9 | HTML report / model card output | A | VPR readiness |
| 10 | Suite / catalog / grader versioning | C | Entire governance layer |
| 11 | Prometheus `/metrics` endpoint | D | Ops observability |
| 12 | Batch `/verify` endpoint | E | Training throughput |
| 13 | Periodic eval during training (trigger) | A | Phase 2 |
| 14 | Environment test harness (`--strict` validate) | B | Quality assurance |

## What Works Well (keep these)

| Strength | Why it matters |
|----------|---------------|
| `seed()` / `verify()` contract | Simple, obvious, hard to misuse |
| Progress bar with live metrics | Researchers can see if a run is working without waiting hours |
| Full artifact suite (trajectories, runtime, failures) | Decision-grade richness actually delivered |
| Sharding via env vars | Same abstraction across SLURM, K8s, Ray, Docker -- zero lock-in |
| `nel validate` for environment authors | 2-minute feedback loop on new benchmarks |
| `--gym-compat` flag | Zero-friction Gym integration |
| Regression comparison library | Statistical rigor (CI overlap, per-category deltas) |

---

## Recommendation Priority

1. **Fix the three critical bugs** -- these are embarrassing and will be hit within 5 minutes of anyone trying the tool
2. **Add multi-task config execution** -- this is the core Persona A promise and the RFC's centerpiece
3. **Expose `nel regression`** -- one-liner that unlocks the most common post-eval workflow
4. **Add `ray` to optional deps, fix example imports** -- housekeeping that prevents "broken on install" perception
5. **Everything else is Phase 2+** and can be sequenced based on which persona matters most right now
