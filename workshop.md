# NeMo Evaluator Workshop

**Duration:** 35 minutes | **Level:** Beginner to intermediate

## What NeMo Evaluator Does

NeMo Evaluator is a unified LLM evaluation framework. It runs benchmarks
against any model endpoint and produces **decision-grade evaluation evidence**:
scores with confidence intervals, pass@k across repeated runs, per-category
breakdowns, statistical regression detection, and multi-format reports.

The core design has five stages:

```
seed(idx) --> solve(task) --> verify(response) --> score --> decide
  ^              ^                ^                 ^          ^
  |              |                |                 |          |
  Environment    Solver           Environment       Scoring    Decision-grade
  (any source)   (any model)      (same env)        primitives layer
```

An **Environment** provides the dataset and scoring logic. A **Solver** calls
the model. Evaluator drives the loop: it seeds each problem, sends it to
the model, verifies the response, aggregates statistics, and writes artifacts.

All benchmark sources -- built-in, lm-eval, Gym, Prime Intellect,
MTEB, legacy containers, or your own file -- implement the same
`EvalEnvironment` base class.

---

## Setup (2 min)

```bash
git clone ssh://git@gitlab-master.nvidia.com:12051/dl/JoC/competitive_evaluation/nemo-evaluator-next.git nemo-evaluator-next
cd nemo-evaluator-next
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[scoring,stats]"

export NEMO_MODEL_URL=https://inference-api.nvidia.com/v1/chat/completions
export NEMO_MODEL_ID=azure/openai/gpt-5.2
export NEMO_API_KEY=<provided at the workshop>

nel list --source builtin   # 12 built-in benchmarks
```

---

## 1. Evaluator Owns the Loop (5 min)

Run a single benchmark against your model endpoint. Evaluator fetches each
problem, sends it to the model, scores the response, and repeats the process
twice per problem to compute pass@k:

```bash
nel eval run --bench gsm8k --repeats 2 --max-problems 20 \
  -o ./results/01_quick
```

The output directory contains a full evidence bundle -- not just a score:

```bash
python3 -c '
import json, glob
b = json.load(open(glob.glob("./results/01_quick/gsm8k/eval-*.json")[0]))
for m, v in b["benchmark"]["scores"].items():
    if isinstance(v, dict) and "value" in v:
        val, lo, hi = v["value"], v["ci_lower"], v["ci_upper"]
        print(f"  {m}: {val:.4f}  [{lo:.4f}, {hi:.4f}]")
'
```

`pass@1` and `pass@2` with 95% confidence intervals. Plus per-problem
`results.jsonl` (every prompt, response, reward, extracted answer),
`trajectories.jsonl` (timing, tokens, scoring details), and
`runtime_stats.json` (latency percentiles, throughput).

Scale to a suite with one YAML file:

```yaml
# workshop_suite.yaml
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
    max_problems: 20
  - name: mmlu
    max_problems: 20

output:
  dir: ./results/01_suite
  report: [markdown, html]
```

```bash
nel eval run workshop_suite.yaml
nel eval report ./results/01_suite --all-formats
open ./results/01_suite/report.html
```

Override anything from the CLI without editing the file:

```bash
nel eval run workshop_suite.yaml -O benchmarks.0.max_problems=5
```

Every inference response and scoring result is persisted to disk as it
completes. If a run crashes or is interrupted, `--resume` picks up from the
last completed step -- already-scored problems are skipped, cached inference
without scoring is re-verified without calling the model again:

```bash
nel eval run workshop_suite.yaml --resume
```

---

## 2. Everything Is an EvalEnvironment (5 min)

Multiple URI schemes. One base class. One eval loop.

### BYOB in 15 lines

Create `workshop_trivia.py`:

```python
from nemo_evaluator import benchmark, scorer, ScorerInput, fuzzy_match

DATASET = [
    {"question": "What is the capital of France?", "answer": "Paris"},
    {"question": "What is 7 * 8?", "answer": "56"},
    {"question": "Who wrote Romeo and Juliet?", "answer": "William Shakespeare"},
    {"question": "What is the chemical symbol for gold?", "answer": "Au"},
    {"question": "In what year did World War II end?", "answer": "1945"},
]

@benchmark(
    name="workshop_trivia",
    dataset=lambda: DATASET,
    prompt="Answer concisely.\n\nQuestion: {question}\nAnswer:",
    target_field="answer",
)
@scorer
def score(sample: ScorerInput) -> dict:
    return fuzzy_match(sample)
```

Point any CLI command at the `.py` file -- no package installation or
registration step needed:

```bash
nel eval run --bench ./workshop_trivia.py --repeats 2 -o ./results/02_byob
nel serve -b ./workshop_trivia.py -p 9090   # serve as HTTP endpoint for Gym training
nel validate -b ./workshop_trivia.py --samples 3   # dry-run sanity check
```

All 12 built-in benchmarks use the same `@benchmark + @scorer` API with
different scoring primitives: GSM8K uses `numeric_match` (extract last number),
MMLU uses `multichoice_regex` (extract A/B/C/D letter), HumanEval uses
`code_sandbox` (run code in Docker), and SimpleQA uses `needs_judge` (send
response to a second LLM for factual evaluation).

### All environment types in one suite

```bash
pip install lm_eval langdetect immutabledict    # for lm-eval://
```

Start a Gym server so the suite can evaluate against it:

```bash
nel serve -b gsm8k -p 9090 &
sleep 3
```

Create `workshop_mixed.yaml`:

```yaml
model:
  url: ${NEMO_MODEL_URL}
  id: ${NEMO_MODEL_ID}
  api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: gsm8k                   # built-in (numeric_match)
    max_problems: 10
  - name: ./workshop_trivia.py    # BYOB file (fuzzy_match)
    repeats: 2
  - name: lm-eval://ifeval        # lm-evaluation-harness
    max_problems: 10
  - name: gym://localhost:9090     # remote Gym server
    max_problems: 10

output:
  dir: ./results/02_mixed
  report: [markdown, html]
```

```bash
nel eval run workshop_mixed.yaml
kill %1   # stop the Gym server
```

All four benchmarks go through the same seed/solve/verify loop and land in
a single unified report.

---

## 3. Gym Bridge: Serve Benchmarks for Training (2 min)

`nel serve` exposes any benchmark as an HTTP server that speaks the NeMo Gym
REST protocol (`/seed_session`, `/verify`, `/dataset_size`, `/health`). This
means a benchmark you define for evaluation can also be used as a reward
environment for RL training -- without maintaining two implementations.

```bash
nel serve -b gsm8k -p 9090 --gym-compat &
sleep 2

# Gym training calls these endpoints to get problems and score rollouts:
curl -s http://localhost:9090/seed_session \
  -H 'Content-Type: application/json' -d '{"idx":0}'
curl -s http://localhost:9090/verify \
  -H 'Content-Type: application/json' -d '{"response":"42","expected_answer":"42"}'

# Evaluator can also evaluate against the same running server:
nel eval run --bench gym://localhost:9090 --max-problems 10 -o ./results/03_gym

kill %1
```

---

## 4. Regression Detection (2 min)

```bash
nel eval run --bench gsm8k --repeats 3 --max-problems 30 --temperature 0.0 \
  -o ./results/04_baseline

nel eval run --bench gsm8k --repeats 3 --max-problems 30 --temperature 0.7 \
  -o ./results/04_candidate

nel regression \
  ./results/04_baseline/gsm8k/eval-*.json \
  ./results/04_candidate/gsm8k/eval-*.json \
  --strict --threshold 0.05
```

```
! pass@1: 0.7200 -> 0.6800  (delta=-0.0400, -5.6%, CI overlap)

REGRESSIONS detected (>5% drop): pass@1
```

`nel regression` compares the per-problem reward vectors. Metrics prefixed with
`!` regress beyond the threshold. It also checks confidence interval overlap
and computes a Mann-Whitney U p-value when `scipy` is installed. The `--strict`
flag makes the command return a non-zero exit code when any metric regresses
beyond `--threshold`, so you can use it as a CI gate -- for example, blocking a
merge request when evaluation scores drop.

---

## 5. Advanced Scoring (3 min)

### LLM-as-judge

Some benchmarks can't be scored deterministically -- factual accuracy requires
judgment. SimpleQA and HealthBench use the `needs_judge` scorer, which sends
the model's response to a second LLM (the "judge") that evaluates correctness.
The `services` block lets you configure separate endpoints for the evaluated
model and the judge:

```yaml
# workshop_judge.yaml
services:
  evaluated:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}
  judge:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: simpleqa
    model: evaluated
    judge: judge
    max_problems: 10

output:
  dir: ./results/05_judge
```

```bash
nel eval run workshop_judge.yaml
```

In production you'd use a stronger model as the judge (e.g., GPT-5.2 judging
a fine-tuned Llama). For this workshop we use the same endpoint for both.

### Code sandbox (HumanEval)

HumanEval evaluates code generation. For each problem, the model writes a
Python function, and NEL executes it with unit tests inside a Docker container
(no network access, 30s timeout, 256MB memory limit):

```bash
nel eval run --bench humaneval --max-problems 10 --temperature 0.2 \
  -o ./results/05_code
```

> Requires Docker running locally.

---

## 6. Agentic Evaluation (5 min)

NeMo Evaluator supports evaluating autonomous agents -- programs that receive
a task prompt and interact with tools, sandboxes, and APIs to produce a
solution. The evaluator provides the infrastructure: per-problem Docker
sandboxes, task delivery, and verification.

### How it works

Set `endpoint_type: sandbox` on a benchmark and configure the agent command.
For each problem, the eval loop:

1. Seeds the problem from the environment (prompt + optional sandbox spec)
2. Starts a Docker sandbox from the per-problem image
3. Runs the agent command inside the sandbox
4. The agent interacts with the codebase using `$MODEL_BASE_URL` to reach
   the model
5. The environment's `verify()` checks the sandbox state (e.g., runs tests)

### Harbor + SWE-Bench

Harbor provides agent-centric benchmarks like SWE-Bench. Each problem has
its own Docker image with a code repository and a bug to fix:

```yaml
# workshop_harbor_agent.yaml
model:
  url: ${NEMO_MODEL_URL}
  id: ${NEMO_MODEL_ID}
  api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: harbor://swebench-verified
    endpoint_type: sandbox
    max_problems: 3
    sandbox:
      backend: docker
      agent_cmd: openhands
      agent_setup_cmd: "pip install openhands-ai"
      agent_invocation_template: >-
        openhands --task-file {task_file}
        --output-file {output_file}
        --model-base-url {model_url}
        --model-id {model_id}
      concurrency: 2
      timeout: 1800

output:
  dir: ./results/06_harbor_agent
```

> Harbor requires Docker, the Harbor datasets directory, and the agent binary
> installed in the sandbox image. This example is shown for reference -- see
> the Harbor README for full setup instructions.

### BYOB + Agent with sandbox-aware scoring

You can build fully custom agent evaluation pipelines using BYOB. The scorer
function is `async` and gets direct access to the sandbox the agent just
modified:

```python
from nemo_evaluator import benchmark, scorer, ScorerInput
from nemo_evaluator.environments.base import SeedResult, SandboxSpec

TASKS = [
    {"instruction": "Create a file called hello.txt with 'Hello World'",
     "check_cmd": "cat /workspace/hello.txt"},
]

def my_seed(row, idx):
    return SeedResult(
        prompt=row["instruction"],
        expected_answer="",
        sandbox_spec=SandboxSpec(image="python:3.12-slim"),
        metadata={"check_cmd": row["check_cmd"]},
    )

@benchmark(name="agent_demo", dataset=lambda: TASKS,
           prompt="", seed_fn=my_seed)
@scorer
async def score(sample: ScorerInput) -> dict:
    if sample.sandbox is None:
        return {"correct": 0}
    result = await sample.sandbox.exec(sample.metadata["check_cmd"])
    return {"correct": 1 if result.return_code == 0 else 0}
```

---

## 7. PinchBench + NAT Agent (5 min)

PinchBench is an agentic task benchmark that tests an agent's ability to
use tools and create workspace artifacts (files, code, etc.).
NeMo Agent Toolkit (NAT) provides a framework for building configurable
agents with tool access.

The integration is fully interchangeable:
- **NAT agents work with any benchmark** (PinchBench, MMLU, Gym...)
- **PinchBench works with any solver** (NatSolver, AgentSolver, ChatSolver...)

### Install NAT

```bash
pip install "nvidia-nat[langchain]"
pip install langchain-nvidia-ai-endpoints
pip install -e ".[scoring,stats]"      # reinstall nel (nvidia-nat may shadow it)
```

> The last line is needed because NAT's dependency tree may overwrite the `nel`
> CLI entry point. Re-installing nemo-evaluator restores it.

### Configure and start a NAT agent

Create `nat_agent.yaml`:

```yaml
llms:
  gpt:
    _type: nim
    model_name: azure/openai/gpt-5.2
    base_url: https://inference-api.nvidia.com/v1
    api_key: ${NEMO_API_KEY}
    temperature: 0.0
    max_tokens: 2048

functions:
  current_datetime:
    _type: current_datetime

workflow:
  _type: react_agent
  tool_names: [current_datetime]
  llm_name: gpt
  verbose: true
```

```bash
nat serve --config_file nat_agent.yaml --host 0.0.0.0 --port 9000 &
sleep 5
curl -s http://localhost:9000/health   # {"status":"healthy"}
```

### NAT agent with a standard benchmark

Before PinchBench, verify the NAT integration works with a simple benchmark.
NatSolver sends the task prompt to NAT's `/generate/full` SSE endpoint and
extracts the final text response:

```yaml
# workshop_nat_gsm8k.yaml
model:
  url: http://localhost:9000
  id: nat-agent

benchmarks:
  - name: gsm8k
    endpoint_type: nat
    max_concurrent: 1
    max_problems: 5

output:
  dir: ./results/07_nat_gsm8k
```

```bash
nel eval run workshop_nat_gsm8k.yaml
```

### NAT agent with PinchBench

PinchBench tasks include both automated scoring (embedded `grade()` functions)
and judge-scored tasks that need LLM-as-judge. The `judge` service uses the
same GPT-5.2 endpoint:

```yaml
# workshop_nat_pinchbench.yaml
services:
  model:
    type: api
    url: http://localhost:9000
    model: nat-agent
  judge:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: pinchbench
    model: model
    judge: judge
    endpoint_type: nat
    max_concurrent: 1
    max_problems: 5

output:
  dir: ./results/07_nat_pinchbench
```

```bash
nel eval run workshop_nat_pinchbench.yaml
```

NatSolver collects the full trajectory (LLM calls, tool invocations, results),
converts it to PinchBench's transcript format, and writes it to the task
workspace. The PinchBench scorer then executes each task's embedded `grade()`
function against the transcript and workspace files. Tasks without a `grade()`
function are evaluated by the judge LLM against the task's grading criteria.

### PinchBench with a regular ChatSolver (no agent)

For comparison, run the same benchmark with a plain ChatSolver. Tasks
requiring tool calls and file creation will score lower without a real agent,
but the judge still evaluates the response against the grading criteria:

```yaml
# workshop_pinchbench_chat.yaml
services:
  model:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}
  judge:
    type: api
    url: ${NEMO_MODEL_URL}
    model: ${NEMO_MODEL_ID}
    api_key: ${NEMO_API_KEY:-}

benchmarks:
  - name: pinchbench
    model: model
    judge: judge
    max_problems: 5

output:
  dir: ./results/07_chat_pinchbench
```

```bash
nel eval run workshop_pinchbench_chat.yaml
kill $(lsof -ti:9000) 2>/dev/null   # stop the NAT server
```

---

## Appendix: Solver x Environment Compatibility

Not every solver works with every environment. See the
[Solver / Environment Compatibility table in README.md](README.md#solver--environment-compatibility)
for the full matrix. The evaluator warns at runtime for known-bad pairings.

---

## Appendix: Running on SLURM

All examples in this workshop use a remote API endpoint. For self-hosted model
evaluation on SLURM clusters, set `cluster.type: slurm` in your config. NEL
generates a self-contained sbatch script that starts model servers inside
containers (via Pyxis/Enroot), runs benchmarks, and collects results. NAT
agents can run as managed SLURM services with `type: nat` and a custom
container image. See the `deploy/` directory for full SLURM examples.

---

## Cleanup

```bash
rm -rf ./results
rm -f workshop_suite.yaml workshop_mixed.yaml
rm -f workshop_judge.yaml workshop_trivia.py
rm -f workshop_nat_pinchbench.yaml workshop_pinchbench_chat.yaml
rm -f workshop_nat_gsm8k.yaml nat_agent.yaml
```
