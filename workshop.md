# NeMo Evaluator Workshop

**Duration:** 60 minutes | **Level:** Beginner to intermediate

NeMo Evaluator (`nel`) is a unified framework for evaluating language models.
It provides 11 built-in benchmarks, integrations with NeMo Skills and
lm-evaluation-harness, a Gym-compatible server, per-problem code sandboxes,
SLURM-native distribution, and a BYOB system for custom benchmarks -- all
behind a single CLI.

```
Config ─→ Executor ─→ Solver ─→ Environment ─→ Scorer ─→ Artifacts
(YAML)    (local/     (chat/    (benchmark     (scoring   (bundle,
           slurm/      vlm/      + dataset)     func)      reports,
           docker)     agent)                              JSONL)
```

**What you will learn:**

1. Run evaluations from the CLI and YAML configs
2. Use NeMo Skills, lm-eval, Gym, and legacy container environments
3. Build and serve your own benchmark (BYOB)
4. Compare runs with statistical regression testing
5. Scale to SLURM clusters
6. Generate multi-format reports

---

## Setup (3 min)

### Clone and install

```bash
git clone https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nemo-evaluator-next.git
cd nemo-evaluator-next

python3 -m venv .venv
source .venv/bin/activate          # On Windows: .venv\Scripts\activate

pip install -e ".[scoring,stats]"
```

> **Python 3.10+ required.** Verify with `python3 --version`.

### Set up the model endpoint

We use NVIDIA Inference API -- no local GPU needed.
The API key will be provided during the workshop.

```bash
export NEMO_MODEL_URL=https://inference-api.nvidia.com/v1/chat/completions
export NEMO_MODEL_ID=azure/openai/gpt-5.2
export NEMO_API_KEY=<API_KEY provided at the workshop>
```

### Verify

```bash
nel --help
nel list --source builtin
```

You should see the 11 built-in benchmarks: `gsm8k`, `mmlu`, `triviaqa`, etc.

> **All commands in this workshop run from the repo root** (`nemo-evaluator-next/`).

---

## Part 1: First Evaluation — CLI Quick Mode (5 min)

Run a single benchmark with zero config:

```bash
nel eval run --bench gsm8k \
  --repeats 2 --max-problems 20 \
  -o ./workshop_results/01_quick
```

Watch the live progress bar. When it finishes, explore what was produced:

```bash
ls ./workshop_results/01_quick/gsm8k/
```

| File | Contents |
|------|----------|
| `eval-*.json` | Full bundle: scores, config, confidence intervals |
| `results.jsonl` | Per-problem: prompt, response, reward, extracted answer |
| `trajectories.jsonl` | Step records: timing, tokens, scoring details |
| `runtime_stats.json` | Latency percentiles, throughput |

Inspect the scores:

```bash
python3 -c '
import json, glob
b = json.load(open(glob.glob("./workshop_results/01_quick/gsm8k/eval-*.json")[0]))
for m, v in b["benchmark"]["scores"].items():
    if isinstance(v, dict) and "value" in v:
        print(f"  {m}: {v["value"]:.4f}  [{v["ci_lower"]:.4f}, {v["ci_upper"]:.4f}]")
'
```

**Checkpoint:** You see `pass@1` and `pass@2` with 95% confidence intervals.

---

## Part 2: Multi-Benchmark Suite via Config (7 min)

Create `workshop_suite.yaml` in the repo root:

```yaml
model:
  url: ${NEMO_MODEL_URL}
  id: ${NEMO_MODEL_ID}
  api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: gsm8k
    repeats: 2
    max_problems: 20
    system_prompt: "Solve step by step. Put your final answer after 'The answer is'."

  - name: triviaqa
    repeats: 1
    max_problems: 20

  - name: mmlu
    repeats: 1
    max_problems: 20

output:
  dir: ./workshop_results/02_suite
  report:
    - markdown
    - html
```

> **Env var expansion:** `${VAR}` and `${VAR:-default}` work in any string field.
> This means your config is portable -- no hardcoded secrets.

```bash
nel eval run workshop_suite.yaml
```

Each benchmark runs sequentially with automatic checkpointing. Override any field
from the CLI without editing the file:

```bash
nel eval run workshop_suite.yaml -O benchmarks.0.max_problems=5
```

Generate reports from the results:

```bash
nel eval report ./workshop_results/02_suite --all-formats -o ./workshop_results/02_suite/report
```

Open `report.html` in your browser. MMLU shows per-subject category breakdowns.

### Resume after failure

NEL persists every step incrementally to disk. If a run crashes, no work is lost.

**Step 1 — start a long run and interrupt it:**

```bash
nel eval run --bench gsm8k --repeats 3 --max-problems 20 \
  -o ./workshop_results/02_resume

# While running, press Ctrl+C after a few seconds to simulate a crash
```

**Step 2 — inspect what was saved:**

```bash
wc -l ./workshop_results/02_resume/gsm8k/inference_log.jsonl
wc -l ./workshop_results/02_resume/gsm8k/verified_log.jsonl
```

You'll see partial results: inference responses and scoring results persisted
per-step as they completed.

**Step 3 — resume and observe skipping:**

```bash
nel eval run --bench gsm8k --repeats 3 --max-problems 20 \
  -o ./workshop_results/02_resume --resume
```

Only the remaining steps execute. Already-verified steps are skipped entirely;
steps with cached inference but no scoring are re-verified without calling the
model again.

**Re-verify trick:** If you change your scorer or judge config, delete
`verified_log.jsonl` and resume — inference is reused, only scoring re-runs:

```bash
rm ./workshop_results/02_resume/gsm8k/verified_log.jsonl
nel eval run --bench gsm8k --repeats 3 --max-problems 20 \
  -o ./workshop_results/02_resume --resume
```

**Checkpoint:** Resume also works at suite level — completed benchmarks are
skipped, partial benchmarks pick up where they left off.

The same `--resume` flag works with config YAML files:

```bash
nel eval run workshop_suite.yaml --resume
```

---

## Part 3: External Environments (8 min)

NEL resolves benchmarks by URI scheme. Every scheme below works in both
`--bench` and YAML configs.

### 3a. NeMo Skills (`skills://`)

Requires: `pip install nemo-skills` (or `pip install -e ".[skills]"`)

```bash
nel eval run --bench skills://gsm8k \
  --repeats 1 --max-problems 15 \
  -o ./workshop_results/03_skills
```

Skills environments use NeMo Skills' own graders (math_equal, multichoice),
which can be stricter or more lenient than built-in scorers.

### 3b. lm-evaluation-harness (`lm-eval://`)

Requires: `pip install lm_eval` (or `pip install -e ".[lm-eval]"`)

```bash
nel eval run --bench lm-eval://aime2025 \
  --repeats 1 --max-problems 10 \
  -o ./workshop_results/03_lmeval
```

### 3c. Gym environments (`gym://`)

Evaluate against any running Gym server:

```bash
# Start a server (in background)
nel serve -b gsm8k -p 9090 &
sleep 2

# Evaluate against it
nel eval run --bench gym://localhost:9090 \
  --repeats 2 --max-problems 15 \
  -o ./workshop_results/03_gym

# Clean up
kill %1
```

This is the same protocol NeMo Gym training uses. The model call goes through
NEL (full observability), while seed/verify go through the Gym server.

### 3d. Legacy containers (`container://`)

For eval-factory containers that own the entire eval loop:

```yaml
benchmarks:
  - name: container://nvcr.io/eval-factory/mmlu:latest#mmlu_pro
```

The container runs via `docker run`, and NEL parses the output into its
standard artifact format. Useful for migrating existing container-based evals.

### 3e. Mix everything in one config

```yaml
# workshop_mixed.yaml
model:
  url: ${NEMO_MODEL_URL}
  id: ${NEMO_MODEL_ID}
  api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: gsm8k                # built-in
    max_problems: 10
  - name: skills://mmlu-pro     # NeMo Skills
    max_problems: 10
  - name: lm-eval://aime2025    # lm-evaluation-harness
    max_problems: 5

output:
  dir: ./workshop_results/03_mixed
  report: [markdown]
```

```bash
nel eval run workshop_mixed.yaml
```

**Checkpoint:** Three URI schemes resolve and evaluate in one suite.

---

## Part 4: Build Your Own Benchmark — BYOB (10 min)

A BYOB benchmark is a standalone Python file. No need to install it inside the
package or edit any config -- just point `nel` at the file.

### 4a. Create the benchmark file

Create `workshop_trivia.py` anywhere (e.g. in the repo root):

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, fuzzy_match

DATASET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "In what year did World War II end?", "answer": "1945"},
    {"question": "What planet is closest to the Sun?", "answer": "Mercury"},
    {"question": "What is the square root of 144?", "answer": "12"},
    {"question": "What gas do plants absorb?", "answer": "carbon dioxide"},
    {"question": "What is the largest ocean on Earth?", "answer": "Pacific Ocean"},
    {"question": "Who painted the Mona Lisa?", "answer": "Leonardo da Vinci"},
]


@benchmark(
    name="workshop_trivia",
    dataset=lambda: DATASET,
    prompt="Answer concisely in a few words.\n\nQuestion: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def workshop_trivia_scorer(sample: ScorerInput) -> dict:
    return fuzzy_match(sample)
```

That's the entire benchmark. The `@benchmark` decorator handles dataset loading,
prompt formatting, and registration. The `@scorer` function defines scoring.

### 4b. Run it directly from the file

Point `--bench` at the `.py` file -- NEL imports it automatically:

```bash
nel eval run --bench ./workshop_trivia.py --repeats 3 -o ./workshop_results/04_byob
```

No registration step, no package edits. All CLI commands accept `.py` paths:

```bash
# Validate
nel validate -b ./workshop_trivia.py --samples 5

# Serve for Gym training
nel serve -b ./workshop_trivia.py -p 9090
```

### 4c. Use in YAML configs

File paths work in config files too:

```yaml
model:
  url: ${NEMO_MODEL_URL}
  id: ${NEMO_MODEL_ID}
  api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: ./workshop_trivia.py
    repeats: 3
  - name: gsm8k                   # mix with built-in benchmarks
    max_problems: 10

output:
  dir: ./workshop_results/04_mixed
```

### 4d. Use a HuggingFace dataset instead of inline data

Replace the `dataset` field to load from HuggingFace:

```python
@benchmark(
    name="my_hf_bench",
    dataset="hf://your-org/your-dataset?split=test",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
```

Or a local JSONL file:

```python
@benchmark(
    name="my_local_bench",
    dataset="data/my_problems.jsonl",
    prompt="Question: {question}\nAnswer:",
    target_field="answer",
)
```

### 4e. Advanced: if the file defines multiple benchmarks

If a single file registers multiple benchmarks, select one explicitly:

```bash
nel eval run --bench ./my_benchmarks.py:specific_name --repeats 2
```

### 4f. Scoring primitives reference

Your `@scorer` function can use any of these:

| Function | When to use | Example benchmark |
|----------|-------------|-------------------|
| `exact_match` | Normalized string equality | custom QA |
| `fuzzy_match` | Substring containment | TriviaQA |
| `multichoice_regex` | Extract A/B/C/D letter | MMLU |
| `answer_line` | Extract text after "Answer:" | MATH-500 |
| `numeric_match` | Last number in response | GSM8K |
| `code_sandbox` | Docker-sandboxed test execution | HumanEval |
| `needs_judge` | LLM-as-judge post-processing | SimpleQA |

Composing scorers (fallback chain):

```python
@scorer
def my_scorer(sample: ScorerInput) -> dict:
    result = answer_line(sample)
    if result["correct"]:
        return result
    return numeric_match(sample)  # fallback
```

**Checkpoint:** Custom benchmark created, validated, evaluated, and served over HTTP.

---

## Part 5: Regression Testing (5 min)

### 5a. Run baseline and candidate

```bash
# Baseline: temperature=0
nel eval run --bench gsm8k \
  --repeats 3 --max-problems 30 --temperature 0.0 \
  -o ./workshop_results/05_regression/baseline

# Candidate: temperature=0.7
nel eval run --bench gsm8k \
  --repeats 3 --max-problems 30 --temperature 0.7 \
  -o ./workshop_results/05_regression/candidate
```

### 5b. Compare

```bash
nel regression \
  ./workshop_results/05_regression/baseline/gsm8k/eval-*.json \
  ./workshop_results/05_regression/candidate/gsm8k/eval-*.json \
  --strict --threshold 0.05
```

Output:

```
  pass@1: 0.7200 -> 0.6800  (delta=-0.0400, -5.6%, p=0.0312 *)
```

Deltas marked `*` are statistically significant (Mann-Whitney U, p < 0.05).
With `--strict`, exit code is non-zero if any score drops beyond the threshold.

Save the report:

```bash
nel regression \
  ./workshop_results/05_regression/baseline/gsm8k/eval-*.json \
  ./workshop_results/05_regression/candidate/gsm8k/eval-*.json \
  --output ./workshop_results/05_regression/regression.json
```

This is designed for CI pipelines: run baseline and candidate in parallel,
then gate the merge on `nel regression --strict`.

**Checkpoint:** Score deltas with p-values. Non-zero exit on regression.

---

## Part 6: Advanced Services and Judge Models (5 min)

The `services` block replaces `model` for advanced configs with multiple
endpoints or LLM-as-judge scoring.

Create `workshop_judge.yaml`:

```yaml
services:
  evaluated:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}

  judge:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}      # ideally a stronger model
    api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: simpleqa
    model: evaluated
    judge: judge
    max_problems: 10

  - name: gsm8k
    model: evaluated
    max_problems: 15

output:
  dir: ./workshop_results/06_judge
  report: [markdown, html]
```

```bash
nel eval run workshop_judge.yaml
```

SimpleQA uses `needs_judge` scoring -- the judge model evaluates factual
correctness of the response. GSM8K uses deterministic scoring (no judge needed).
Both coexist in one suite.

**Checkpoint:** Judge-scored and deterministic benchmarks in one run.

---

## Part 7: SLURM Cluster Deployment (10 min)

### 7a. Local dry-run (generates scripts without submitting)

Create `workshop_slurm.yaml`:

```yaml
services:
  evaluated:
    type: vllm
    model: Qwen/Qwen3.5-9B
    tensor_parallel_size: 8
    port: 8000
    extra_args: ["--max-model-len", "32768"]

benchmarks:
  - name: gsm8k
    model: evaluated
    repeats: 5
    system_prompt: "Solve step by step. Put your final answer after 'The answer is'."

  - name: mmlu
    model: evaluated
    repeats: 1

cluster:
  type: slurm
  partition: batch
  account: coreai_dlalgo_compeval
  nodes: 1
  ntasks_per_node: 1
  gres: "gpu:8"
  walltime: "01:00:00"
  conda_env: gym

output:
  dir: ./eval_results/slurm_workshop
  report: [markdown, html, csv]
```

Generate the sbatch script without submitting:

```bash
nel eval run workshop_slurm.yaml --dry-run
```

Inspect the generated script:

```bash
cat ./eval_results/slurm_workshop/nel_eval.sbatch
```

The script:
1. Activates the conda env
2. Starts vLLM with the model + tensor parallelism
3. Waits for the health check
4. Runs each benchmark sequentially
5. Tears down the model server
6. Writes artifacts and reports

> **Container images**: CI builds evaluator Docker images automatically and
> pushes them to the GitLab container registry at
> `gitlab-master.nvidia.com:5005/dl/JoC/competitive_evaluation/nemo-evaluator-next`.
> Variants: `:latest` (base), `:latest-lm-eval`, `:latest-skills`, `:latest-mteb`, `:latest-full`.
>
> NEL picks the correct variant per benchmark automatically. To use a
> different registry or tag, set `container_image` to your base image —
> variants are derived by appending `-lm-eval`, `-skills`, etc. to the tag:
>
> ```yaml
> cluster:
>   container_image: my-registry.io/nel:v2.0
>   # → lm-eval benchmarks use my-registry.io/nel:v2.0-lm-eval
>   # → skills benchmarks use my-registry.io/nel:v2.0-skills
> ```

### 7b. Submit to a cluster

```bash
nel eval run workshop_slurm.yaml --submit
```

Or submit over SSH to a remote login node:

```yaml
cluster:
  type: slurm
  hostname: cw-dfw-cs-001-login-01
  username: ${USER}
  partition: batch
  account: coreai_dlalgo_compeval
  # ... rest of config
```

### 7c. Monitor and control

```bash
nel eval status -o ./eval_results/slurm_workshop
nel eval stop -o ./eval_results/slurm_workshop
```

### 7d. SLURM with Gym environment

```yaml
services:
  evaluated:
    type: vllm
    model: Qwen/Qwen3.5-9B
    tensor_parallel_size: 8
    port: 8000

  swebench_gym:
    type: gym
    benchmark: swebench
    port: 9090

benchmarks:
  - name: gsm8k
    model: evaluated
    repeats: 5

  - name: gym://localhost:9090
    model: evaluated
    repeats: 1

cluster:
  type: slurm
  partition: batch
  gres: "gpu:8"
  walltime: "02:00:00"
```

The generated sbatch script starts both the model server and the Gym server,
waits for health checks, and orchestrates the evaluations.

### 7e. SLURM with legacy containers

For environments that ship as Docker images (eval-factory pattern):

```yaml
benchmarks:
  - name: container://nvcr.io/eval-factory/mmlu:latest#mmlu_pro

cluster:
  type: slurm
  partition: batch
  gres: "gpu:8"
  walltime: "01:00:00"
  container_image: nvcr.io/eval-factory/mmlu:latest
  container_mounts: "/data:/data:ro"
```

### 7f. Distributed sharding

For large datasets, NEL shards automatically across SLURM array jobs:

```bash
# Manual sharding (any orchestrator):
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel eval run --bench mmlu -o ./results/shard_0
NEL_SHARD_IDX=1 NEL_TOTAL_SHARDS=4 nel eval run --bench mmlu -o ./results/shard_1
# ... (run in parallel across nodes)
```

On SLURM, `SLURM_ARRAY_TASK_ID` and `SLURM_ARRAY_TASK_COUNT` are detected
automatically -- just set `shards` in the cluster config.

**Checkpoint:** You can generate, inspect, and submit SLURM jobs for any
combination of model servers, Gym environments, and legacy containers.

---

## Part 8: Gym Integration Deep Dive (5 min)

### Serve any benchmark for Gym training

```bash
nel serve -b gsm8k -p 9090
```

The server speaks Gym's native protocol:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/dataset_size` | POST | Number of problems |
| `/seed_session` | POST | Get prompt + expected answer |
| `/verify` | POST | Score a response |

Point your Gym training config at it:

```yaml
resource_servers:
  nemo_evaluator:
    endpoint: http://evaluator-host:9090
```

### Export data for batch rollout collection

```bash
nel serve -b gsm8k --export-data /tmp/rollout_data
```

Produces JSONL files compatible with `ng_collect_rollouts`.

### Evaluate against a remote Gym environment

```bash
nel eval run --bench gym://remote-host:9090 --repeats 4
```

**Checkpoint:** Same benchmark serves training and produces decision-grade
evaluation artifacts.

---

## Part 9: Reports, Background Runs, Discovery (2 min)

### Multi-format reports

```bash
nel eval report ./workshop_results/02_suite -f html -o report.html
nel eval report ./workshop_results/02_suite -f latex -o table.tex
nel eval report ./workshop_results/02_suite --all-formats -o report
```

Formats: `markdown`, `html`, `csv`, `json`, `latex`.

### Background runs

```bash
nel eval run --bench triviaqa --max-problems 10 -o /tmp/bg_test --background
nel eval status -o /tmp/bg_test
nel eval stop -o /tmp/bg_test
```

### Benchmark discovery

```bash
nel list --source builtin       # 11 built-in benchmarks
nel list --source skills        # NeMo Skills catalogue
nel list --source lm-eval       # lm-evaluation-harness tasks
nel list --source all           # everything
nel validate -b gsm8k --samples 3   # dry-run sanity check
```

---

## Cleanup

```bash
rm -rf ./workshop_results /tmp/bg_test /tmp/rollout_data
rm -f workshop_suite.yaml workshop_mixed.yaml workshop_judge.yaml workshop_slurm.yaml
rm -f workshop_trivia.py
```

---

## Quick Reference

### CLI commands

| Task | Command |
|------|---------|
| Run one benchmark | `nel eval run --bench gsm8k --repeats 2` |
| Run from config | `nel eval run config.yaml` |
| Override config | `nel eval run config.yaml -O model.temperature=0.5` |
| Resume (step-level) | `nel eval run config.yaml --resume` |
| Re-verify only | Delete `verified_log.jsonl`, then `--resume` |
| Check status | `nel eval status -o ./results` |
| Stop | `nel eval stop -o ./results` |
| Reports | `nel eval report ./results --all-formats` |
| Compare runs | `nel regression baseline/eval-*.json candidate/eval-*.json --strict` |
| List benchmarks | `nel list --source all` |
| Validate | `nel validate -b gsm8k --samples 5` |
| Serve for Gym | `nel serve -b gsm8k -p 9090` |
| Evaluate remote | `nel eval run --bench gym://host:9090` |
| SLURM dry-run | `nel eval run slurm.yaml --dry-run` |
| SLURM submit | `nel eval run slurm.yaml --submit` |
| Package BYOB | `nel package -m my_bench.py --push` |

### URI schemes

| Scheme | Example | Source |
|--------|---------|--------|
| (bare name) | `gsm8k` | Built-in (11 benchmarks) |
| `skills://` | `skills://mmlu-pro` | NeMo Skills |
| `lm-eval://` | `lm-eval://aime2025` | lm-evaluation-harness |
| `gym://` | `gym://host:9090` | Remote Gym server |
| `container://` | `container://image:tag#task` | Legacy eval-factory containers |
| `mteb://` | `mteb://STSBenchmark` | MTEB embedding benchmarks |
| `pi://` | `pi://simpleqa` | Prime Intellect verifiers |

### Scoring primitives

| Function | Use case | Example |
|----------|----------|---------|
| `exact_match` | Normalized string equality | custom QA |
| `fuzzy_match` | Substring containment | TriviaQA |
| `multichoice_regex` | Extract A/B/C/D letter | MMLU |
| `answer_line` | Extract text after "Answer:" | MATH-500 |
| `numeric_match` | Last number in response | GSM8K |
| `code_sandbox` | Docker-sandboxed test execution | HumanEval |
| `needs_judge` | LLM-as-judge post-processing | SimpleQA |

### Config structure

```yaml
# Simple mode
model:
  url: https://inference-api.nvidia.com/v1/chat/completions
  id: azure/openai/gpt-5.2
  api_key: ${NEMO_API_KEY}

# Advanced mode (replaces model:)
services:
  evaluated:
    type: vllm | sglang | nim | api
    model: Qwen/Qwen3.5-9B
    port: 8000
  judge:
    type: api
    url: https://...

benchmarks:
  - name: gsm8k | skills://name | lm-eval://task | gym://host:port
    model: evaluated        # which service to use
    judge: judge            # for needs_judge benchmarks
    repeats: 4
    max_problems: 100
    system_prompt: "..."
    temperature: 0.0
    endpoint_type: chat | completions | vlm | embedding

cluster:
  type: local | slurm | docker
  partition: batch
  gres: "gpu:8"
  walltime: "02:00:00"

output:
  dir: ./eval_results
  report: [markdown, html, csv, json, latex]
  export: wandb | mlflow
```
