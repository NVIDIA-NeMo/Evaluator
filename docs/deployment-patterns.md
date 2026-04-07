# NeMo Evaluator: Deployment Patterns

## Architecture

```{mermaid}
flowchart TB
    subgraph CLI ["nel CLI"]
        run["nel eval run"]
        serve["nel serve"]
        validate["nel validate"]
    end

    subgraph Core ["Evaluator Core"]
        loop["Eval Loop<br/>seed → model → verify"]
        obs["Observability<br/>trajectories · stats · failures"]
        metrics["Metrics<br/>pass@k · CIs · regression"]
    end

    subgraph Envs ["Environment Sources"]
        local["EvalEnvironment<br/>(GSM8K, TriviaQA, BYOB)"]
        gym_a["GymEnvironment<br/>remote seed/verify"]
        vlmkit["VLMEvalKitEnvironment<br/>VLM benchmarks"]
    end

    subgraph Out ["Outputs"]
        traj["trajectories.jsonl"]
        stats["runtime_stats.json"]
        fail["failure_analysis.json"]
        results["results.jsonl"]
        bundle["eval-{id}.json"]
        reg["regression.json"]
    end

    subgraph External ["Gym Pipeline"]
        gymserve["nel serve<br/>(Gym protocol)"]
        ng["ng_collect_rollouts"]
        jsonl["dataset.jsonl"]
    end

    run --> loop
    validate --> loop
    loop --> obs --> Out
    loop --> metrics --> bundle
    loop --- local & gym_a & vlmkit

    serve --> local
    local --> gymserve --> jsonl --> ng
```

---

## Pattern 1: Direct Evaluation (Local Environment)

**Who:** Research / training owner running benchmarks against a model endpoint.

**What:** Evaluator owns the full loop. Seeds problems from a local `EvalEnvironment`, calls the model, verifies, produces all artifacts.

```bash
# Quick validation
nel validate -b gsm8k --samples 10

# Full run with n-repeats
nel eval run --bench gsm8k --repeats 4 --max-problems 100 -o ./results

# From YAML config
nel eval run eval_config.yaml
```

```yaml
# eval_config.yaml
services:
  model:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: gsm8k
    repeats: 4
    max_problems: 100
    solver:
      type: simple
      service: model
```

Multi-benchmark configs support `--resume` for checkpoint-based recovery:

```bash
nel eval run eval_config.yaml --resume
```

```{mermaid}
sequenceDiagram
    participant CLI as nel eval run
    participant Env as EvalEnvironment
    participant Model as Model API
    participant Disk as Artifacts

    loop each problem × n_repeats
        CLI->>Env: seed(idx)
        Env-->>CLI: prompt, expected, metadata
        CLI->>Model: chat(prompt)
        Model-->>CLI: ModelResponse (content, tokens, latency)
        CLI->>Env: verify(response, expected)
        Env-->>CLI: reward, extracted, scoring_details
        CLI->>Disk: record step → trajectories.jsonl
    end
    CLI->>Disk: bundle.json, runtime_stats.json, results.jsonl
```

**Artifacts produced:** All. Full trajectory with request/response bindings, token counts, latency breakdown per phase, scoring details, failure categorization.

**Environments:** Any registered `EvalEnvironment` subclass — GSM8K, TriviaQA, or any BYOB benchmark.

---

## Pattern 2: Serve for Gym Training

**Who:** Environment developer making an Evaluator benchmark available to Gym's RL pipeline.

**What:** An `EvalEnvironment` is exposed as an HTTP service speaking Gym's `/seed_session` + `/verify` protocol. Gym agents consume it during training. Evaluator independently evaluates the same benchmark with full observability.

```bash
# Serve with evaluator protocol (for nel eval run --adapter)
nel serve -b gsm8k --port 9090

# Serve with gym-compatible protocol (accepts NeMoGymResponse in /verify)
nel serve -b gsm8k --port 9090 --gym-compat

# Also export JSONL for ng_collect_rollouts
nel serve -b gsm8k --port 9090 --gym-compat --export-data /data
```

```{mermaid}
sequenceDiagram
    participant Gym as Gym Agent
    participant Svc as nel serve (EvalEnvironment)
    participant Eval as nel eval run (independent)

    Note over Gym,Svc: Training loop
    Gym->>Svc: POST /seed_session
    Svc-->>Gym: {prompt, expected_answer}
    Gym->>Gym: model call
    Gym->>Svc: POST /verify {response: NeMoGymResponse}
    Svc-->>Gym: {reward: 1.0}
    Note over Gym: reads reward only

    Note over Eval,Svc: Independent evaluation
    Eval->>Svc: POST /seed_session {idx}
    Svc-->>Eval: {prompt, expected_answer, metadata}
    Eval->>Eval: model call (captured)
    Eval->>Svc: POST /verify {response, expected}
    Svc-->>Eval: {reward, scoring_details, metadata}
    Note over Eval: full artifact bundle
```

**Dual-consumer:** Same service, two readers. Gym reads `reward`. Evaluator reads everything.

---

## Pattern 3: Consume Remote Environment via Adapter

**Who:** Evaluation owner who wants Evaluator's statistical rigor on an environment running elsewhere.

**What:** `GymEnvironment` connects to any server speaking `seed_session/verify`. Evaluator owns the model call in between, giving full trajectory capture even though the environment is remote.

```bash
# Consume a remote environment
nel eval run --bench gym://staging-server:8080 --repeats 4 --max-problems 50

# Consume another nel serve instance
nel eval run --bench gym://127.0.0.1:9090 --repeats 2
```

```{mermaid}
sequenceDiagram
    participant Eval as nel eval run
    participant Env as GymEnvironment
    participant Remote as Remote Environment
    participant Model as Model API

    loop each problem × n_repeats
        Eval->>Env: seed(idx)
        Env->>Remote: POST /seed_session {idx}
        Remote-->>Env: {prompt, expected_answer}
        Env-->>Eval: SeedResult

        Eval->>Model: chat(prompt)
        Note over Eval: trajectory captured

        Eval->>Env: verify(response, expected)
        Env->>Remote: POST /verify
        Remote-->>Env: {reward, scoring_details}
        Env-->>Eval: VerifyResult
    end
```

**Key:** Evaluator owns the model call → full trajectory, tokens, latency regardless of where the environment runs.

---

## Pattern 4: VLMEvalKit Benchmarks

**Who:** Research team evaluating vision-language models against VLMEvalKit's 100+ benchmarks with Evaluator's full observability.

**What:** `VLMEvalKitEnvironment` wraps VLMEvalKit datasets, handling image loading and scoring (MCQ, VQA, Y/N). The `VLMSolver` sends images + text to the model.

```bash
nel eval run --bench vlmevalkit://MMBench_DEV_EN --repeats 1 --max-problems 50
```

```python
from nemo_evaluator.environments.vlmevalkit import VLMEvalKitEnvironment
from nemo_evaluator.engine import run_evaluation, ModelClient
from nemo_evaluator.solvers import VLMSolver

env = VLMEvalKitEnvironment("MMBench_DEV_EN")
client = ModelClient(base_url="...", model="...", api_key="...")
solver = VLMSolver(client)
bundle = await run_evaluation(env, solver, n_repeats=1)
```

```{mermaid}
sequenceDiagram
    participant Eval as nel eval run
    participant VLM as VLMEvalKitEnvironment
    participant VK as VLMEvalKit Dataset
    participant Model as VLM API

    VLM->>VK: build_dataset("MMBench_DEV_EN")

    loop each problem × n_repeats
        Eval->>VLM: seed(idx)
        VLM->>VK: build_prompt(line)
        VK-->>VLM: images + prompt text
        VLM-->>Eval: SeedResult (prompt, images, choices)

        Eval->>Model: chat(images + prompt)
        Note over Eval: trajectory captured

        Eval->>VLM: verify(response, expected)
        VLM-->>Eval: VerifyResult (MCQ/VQA/Y-N scoring)
    end
```

---

## Pattern 5: Serve for ng_collect_rollouts

**Who:** Training team that needs batch rollout collection using Gym's infrastructure.

**What:** `nel serve` exposes any `EvalEnvironment` as a Gym-compatible HTTP server. Gym agents and `ng_collect_rollouts` consume it via standard `/seed_session` + `/verify` endpoints.

```bash
# Serve gsm8k as a Gym-compatible resource server
nel serve -b gsm8k --gym-compat --port 9090
# Gym training nodes connect via: gym://localhost:9090
```

```python
# Or programmatically
import uvicorn
from nemo_evaluator.environments.registry import get_environment
from nemo_evaluator.serving.app import generate_app

env = get_environment("gsm8k")
app = generate_app(env, gym_compat=True)
uvicorn.run(app, host="0.0.0.0", port=9090)
```

JSONL row format (matches `ng_collect_rollouts` input spec):
```json
{
  "responses_create_params": {
    "input": [{"role": "user", "content": "Solve: ..."}]
  },
  "expected_answer": "42",
  "uuid": "gsm8k-0",
  "eval_type": "gsm8k",
  "metadata": {"category": "math"}
}
```

---

## Pattern 6: Regression Comparison

**Who:** CI pipeline or evaluation owner comparing model versions.

**What:** Two run bundles → score deltas with CI overlap check, Mann-Whitney U p-values, per-category deltas, runtime deltas.

```python
from nemo_evaluator.engine.comparison import compare_runs, write_regression

report = compare_runs("results/v1/eval-*.json", "results/v2/eval-*.json")
write_regression(report, "results/regression.json")
```

Output:
```json
{
  "score_deltas": {
    "pass@1": {"baseline": 0.85, "candidate": 0.88, "delta": 0.03, "ci_overlap": true, "p_value": 0.031, "significant": true}
  },
  "category_deltas": {
    "algebra": {"baseline": 0.92, "candidate": 0.95, "delta": 0.03},
    "geometry": {"baseline": 0.71, "candidate": 0.68, "delta": -0.03}
  },
  "runtime_deltas": {
    "tokens_per_second": {"baseline": 41.0, "candidate": 38.5, "delta": -2.5}
  }
}
```

---

## Artifact Summary

Every evaluation run (all patterns) produces:

| File | Contents |
|------|----------|
| `eval-{id}.json` | Config, pass@k with CIs, per-category scores, runtime stats, failure report |
| `results.jsonl` | Per-sample: problem_idx, repeat, reward, extracted answer, tokens, latency |
| `trajectories.jsonl` | Per-step: full prompt, model response, token breakdown (prompt/completion/reasoning), latency breakdown (seed/model/verify ms), scoring method + details, SHA256 request hash, failure category |
| `runtime_stats.json` | Latency percentiles (p50/p90/p99), token throughput, finish reason distribution, error count |
| `failure_analysis.json` | Categorized failures (refusal, format_error, timeout, rate_limit, empty_response) with counts, percentages, exemplars |
| `regression.json` | Score deltas, CI overlap, per-category and runtime deltas (when comparing two runs) |

---

## Environment Compatibility Matrix

| Source | Evaluator Owns Model Call | Full Trajectory | n_repeats | pass@k + CIs | Progress Bar | Failure Analysis |
|--------|--------------------------|-----------------|-----------|---------------|--------------|------------------|
| Local EvalEnvironment | Yes | Yes | Yes | Yes | Yes | Yes |
| Remote via GymEnvironment | Yes | Yes | Yes | Yes | Yes | Yes |
| VLMEvalKitEnvironment | Yes | Yes | Yes | Yes | Yes | Yes |
| Gym ng_collect_rollouts | No (Gym does) | From output JSONL | Via num_repeats | From reward vectors | Gym's tqdm | Post-hoc |
| Legacy harnesses | No (subprocess) | From output files | Via config | From parsed scores | Process output | Post-hoc |

---

## BYOB: Writing a New Benchmark

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, answer_line

@benchmark(
    name="my_benchmark",
    dataset="hf://my-org/my-data?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    return answer_line(sample)
```

Then:
```bash
nel validate -b my_benchmark --samples 10             # sanity check
nel eval run --bench my_benchmark --repeats 4          # full evaluation
nel serve -b my_benchmark --gym-compat                 # serve for Gym
```

---

## Pattern 7: Distributed Evaluation on SLURM

**User story:** *"I need to evaluate a 14k-problem benchmark with n=8 repeats. Running serially would take days. I want to shard across 16 SLURM nodes and merge results."*

### Architecture

```
                           SLURM Controller
                                 │
                    sbatch --array=0-15
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
               Shard 0      Shard 1  ...  Shard 15
              [p0-p874]    [p875-p1749]  [p13125-p13999]
            nel eval run   nel eval run   nel eval run
                    │            │             │
                    └────────────┼────────────┘
                                 ▼
                          Merge Job
                     (afterok dependency)
                                 │
                                 ▼
                     merged/eval-*.json
                     merged/results.jsonl
                     merged/trajectories.jsonl
```

### Config-Driven SLURM

SLURM evaluations are driven by a YAML config with a `cluster: { type: slurm }` block:

```yaml
# slurm_eval.yaml
services:
  model:
    type: vllm
    model: nvidia/Llama-3.1-70B-Instruct
    protocol: chat_completions
    tensor_parallel_size: 4
    port: 8000
    node_pool: compute

benchmarks:
  - name: gsm8k
    repeats: 8
    solver:
      type: simple
      service: model

cluster:
  type: slurm
  walltime: "04:00:00"
  shards: 16
  node_pools:
    compute:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"

output:
  dir: ./eval_results/gsm8k_distributed
```

```bash
# Submit via config (16 shards, auto-merge after completion)
nel eval run slurm_eval.yaml

# Generate scripts to inspect first (set submit: false in config)
nel eval run slurm_eval.yaml
```

Shard merging is handled automatically when all array tasks complete.

### How Sharding Works

Each SLURM array task gets `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` set automatically. `nel eval run` detects these (or `NEL_SHARD_IDX`/`NEL_TOTAL_SHARDS`) and computes its problem range:

| 14000 problems, 16 shards | Shard 0 | Shard 1 | ... | Shard 15 |
|---|---|---|---|---|
| Problem range | [0, 875) | [875, 1750) | ... | [13125, 14000) |

Each shard writes its own artifacts to `shard_N/`. The merge job combines all results, recomputes global metrics (pass@k, CI), and aggregates runtime stats.

### Serve on SLURM

For long-running Gym training, serve an environment as a SLURM service:

```bash
nel serve -b gsm8k --gym-compat --port 9090

# Wrap in an sbatch script for SLURM submission
# Allocated endpoint is written to eval_results/endpoint.txt
# Gym training nodes connect via: gym://$(cat eval_results/endpoint.txt)
```

### Environment Variables

| Variable | Source | Purpose |
|---|---|---|
| `SLURM_ARRAY_TASK_ID` | SLURM | Shard index (0-based) |
| `SLURM_ARRAY_TASK_COUNT` | SLURM | Total shards |
| `NEL_SHARD_IDX` | Manual override | Same as above, for non-SLURM use |
| `NEL_TOTAL_SHARDS` | Manual override | Same as above |
| `NEMO_API_KEY` | User | Model API key (avoid passing on CLI) |

---

## Pattern 8: Docker / Docker Compose

**User story:** *"I want to run evaluations in containers for reproducibility, or spin up a serve+eval stack locally."*

```bash
# Build the image
docker build -t nemo-evaluator .

# Single eval
docker run -e NEMO_API_KEY=$KEY nemo-evaluator eval run \
    --bench gsm8k --repeats 2 --output-dir /results

# Serve + eval (docker compose)
cd deploy
NEMO_API_KEY=$KEY docker compose up serve eval-remote

# Sharded (manually, 4 shards)
for i in 0 1 2 3; do
  NEL_SHARD_IDX=$i NEL_TOTAL_SHARDS=4 docker compose \
    --profile sharded run -d eval-shard
done
```

Files: `Dockerfile`, `deploy/docker-compose.yaml`

---

## Pattern 9: Kubernetes

**User story:** *"I need to run evaluations on our K8s cluster, with sharded jobs for large benchmarks and a persistent serve endpoint for Gym training."*

### Single Evaluation Job
```bash
kubectl apply -f deploy/k8s/eval-job.yaml
kubectl logs -f job/nel-eval
```

### Sharded (Indexed Job)
Uses K8s `completionMode: Indexed` -- each pod gets `JOB_COMPLETION_INDEX` mapped to `NEL_SHARD_IDX`.

```bash
kubectl apply -f deploy/k8s/eval-indexed-job.yaml
# 8 pods run in parallel, each evaluating its shard
kubectl wait --for=condition=complete job/nel-eval-sharded --timeout=2h
# Then run the merge job
kubectl apply -f deploy/k8s/eval-merge.yaml
```

### Serve as K8s Service
```bash
kubectl apply -f deploy/k8s/serve-deployment.yaml
# Gym training pods connect via: gym://nel-serve.default.svc:9090
```

Includes readiness/liveness probes on `/health`, ClusterIP service for internal discovery.

Files: `deploy/k8s/eval-job.yaml`, `deploy/k8s/eval-indexed-job.yaml`, `deploy/k8s/serve-deployment.yaml`

---

## Pattern 10: Ray (Distributed)

**User story:** *"Our Gym training already runs on Ray. I want to run distributed evaluation on the same Ray cluster."*

```bash
# Submit as a Ray job
ray job submit --working-dir . -- python -m nemo_evaluator.engine.ray_launcher \
    --bench gsm8k --shards 8 --repeats 5 \
    --model-url https://inference-api.nvidia.com/v1 \
    --model-id azure/openai/gpt-5.2 \
    --output-dir ./eval_results/ray

# Or from within a Ray script
import ray
from deploy.ray_eval import run_shard
futures = [run_shard.remote("gsm8k", i, 8, ...) for i in range(8)]
results = ray.get(futures)
```

Each `run_shard` is a Ray remote task that runs `run_evaluation()` with a computed `problem_range`. Results are merged locally after all tasks complete. Works on existing Ray clusters used by Gym training.

File: `src/nemo_evaluator/engine/ray_launcher.py`

---

## Pattern 11: GitLab CI Regression Gate

**User story:** *"I want every MR to automatically evaluate the candidate model against the baseline and block merge if scores regress."*

```yaml
# Include in your .gitlab-ci.yml:
include:
  - local: deploy/gitlab-ci-eval.yml
```

Pipeline stages:
1. **eval:baseline** -- checks out target branch, runs eval
2. **eval:candidate** -- runs eval on MR branch
3. **regression:check** -- compares bundles, fails if any metric drops >5%

Produces `regression.json` artifact with per-metric deltas, CI overlap, and category breakdowns.

File: `deploy/gitlab-ci-eval.yml`

---

## Deployment Matrix

| Target | Eval | Serve | Sharded | Merge | Config |
|---|---|---|---|---|---|
| **Local** | `nel eval run` | `nel serve` | `NEL_SHARD_IDX` env | automatic | CLI flags |
| **SLURM** | `nel eval run config.yaml` | `nel serve` | `--array` | automatic | YAML config |
| **Docker** | `docker run` | `compose up serve` | `compose --profile sharded` | automatic | `docker-compose.yaml` |
| **Kubernetes** | `Job` | `Deployment+Service` | `Indexed Job` | follow-up `Job` | K8s manifests |
| **Ray** | `ray job submit` | N/A (use K8s) | `ray.remote` tasks | in-process | Python |
| **GitLab CI** | pipeline job | N/A | N/A | regression stage | `.gitlab-ci.yml` |

All targets use the same `nel eval run` / `nel serve` commands, same sharding env vars (`NEL_SHARD_IDX`, `NEL_TOTAL_SHARDS`), and same artifact format. The only difference is the orchestration layer.
