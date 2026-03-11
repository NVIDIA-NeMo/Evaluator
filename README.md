# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

## Install

```bash
pip install -e .                # core
pip install -e ".[scoring]"     # + sympy
pip install -e ".[scoring,stats]"  # + sympy + scipy (CIs, regression)
pip install -e ".[mteb]"        # + MTEB embedding benchmarks
pip install -e ".[export]"      # + WandB / MLflow
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
| PinchBench | `nel eval run --bench pinchbench` | Agentic tasks (automated + judge) |

Plus: any lm-eval task (`lm-eval://ifeval`), MTEB embedding benchmark (`mteb://STSBenchmark`), legacy eval-factory container (`container://nvcr.io/image#task`), or remote Gym environment (`gym://host:port`). All external environments use the consistent `scheme://task` URI syntax.

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
| `code_sandbox(sample)` | Docker-sandboxed code execution (sync) |
| `code_sandbox_async(sample, sandbox)` | Sandbox-protocol code execution (async) |
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
| `VLMSolver` | Vision-language models (images + text) |
| `EmbeddingSolver` | Embedding models via `/v1/embeddings` |
| `CrossEncoderSolver` | Reranking via `/v1/rerank` |
| `AgentSolver` | External agent (OpenHands, SWE-agent, etc.) |
| `SandboxedAgentSolver` | Agent in per-problem sandbox (SWE-bench) |
| `NatSolver` | NeMo Agent Toolkit via SSE (`/generate/full`) |

Set `endpoint_type` in your benchmark config to select the solver automatically:

```yaml
benchmarks:
  - name: my-vlm-bench
    endpoint_type: vlm        # chat | completions | vlm | embedding | agent | nat_agent
    image_detail: high
```

## External Harness Dependencies

```bash
pip install lm_eval langdetect immutabledict   # for lm-eval:// environments
```

## NAT Agent Integration

Evaluate NeMo Agent Toolkit agents against any benchmark:

```bash
pip install "nvidia-nat[langchain]" langchain-nvidia-ai-endpoints
pip install -e ".[scoring,stats]"   # reinstall nel (nvidia-nat may shadow the CLI)
```

```yaml
# Start NAT: nat serve --config_file agent.yaml --port 9000
benchmarks:
  - name: pinchbench
    endpoint_type: nat_agent
    max_concurrent: 1
    max_problems: 10
```

NatSolver sends task prompts to NAT's `/generate/full` SSE endpoint, collects the trajectory (LLM calls, tool invocations), and converts it for scoring. Works with any benchmark -- PinchBench, GSM8K, MMLU, etc.

## Resilience

Multi-benchmark suites track per-benchmark completion. A single failure does not abort the suite.

```bash
# Run a suite; if benchmark 2/5 fails, benchmarks 3-5 still execute
nel eval run suite.yaml

# Resume from where you left off (completed benchmarks are skipped)
nel eval run suite.yaml --resume
```

## Per-Problem Sandboxes

For agent evaluations (SWE-bench, terminal-bench) and code execution (HumanEval), NEL provides per-problem isolated sandboxes:

```yaml
benchmarks:
  - name: gym://swebench-server
    sandbox:
      backend: docker                # docker | slurm | local
      image_template: "swebench/sweb.eval.x86_64.{instance_id}:latest"
      concurrency: 8
      memory: 4g
      network: bridge
```

Each problem gets its own container with per-task images. The sandbox stays alive through `seed → solve → verify` so agents can modify files and verification can run tests in the same environment. See [sandbox architecture](docs/architecture/sandbox.md) for details.

## SLURM / Containers

SLURM execution uses Pyxis/Enroot to run each benchmark in a pre-built container. CI builds images automatically and pushes them to the GitLab container registry. Each URI scheme maps to a container variant:

| Tag suffix | Contains |
|------------|----------|
| `:latest` | Base: built-in benchmarks, gym, pi |
| `:latest-lm-eval` | + lm-evaluation-harness |
| `:latest-skills` | + NeMo Skills |
| `:latest-mteb` | + MTEB |
| `:latest-full` | All harnesses |

The sbatch generator automatically selects the right container per benchmark. Two container modes are supported:

- **Colocated** (default): orchestrator + environment in the same container
- **Separated**: each environment runs as a `nel serve` Gym server; orchestrator in base image

```yaml
cluster:
  type: slurm
  env_mode: colocated              # or "separated" for full isolation
  container_image: my-registry.io/nel:v2.0  # optional: overrides default registry
```

## Experiment Tracking

Export results to WandB or MLflow:

```yaml
output:
  dir: ./eval_results
  report: [markdown, html]
  export: [wandb, mlflow]
  export_config:
    wandb:
      project: my-evals
    mlflow:
      experiment_name: nemotron-eval
```

## CLI Reference

| Command | Purpose |
|---------|---------|
| `nel eval run --bench <name>` | Run evaluation |
| `nel eval run config.yaml` | Run from YAML config |
| `nel eval run config.yaml --resume` | Resume a partially completed suite |
| `nel eval run config.yaml -O key=val` | Override config values (dot notation) |
| `nel eval status -o <dir>` | Check running evaluation |
| `nel eval stop -o <dir>` | Stop evaluation |
| `nel eval report <dir>` | Generate reports |
| `nel list` | List available benchmarks |
| `nel serve -b <name>` | Serve benchmark as HTTP endpoint |
| `nel validate -b <name>` | Quick sanity check |
| `nel regression` | Compare two runs (with p-values) |
| `nel config set/get/show/unset` | Persistent user configuration |
| `nel package -m <module> -t <tag>` | Containerize a BYOB benchmark |

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
        mteb.py           MTEB embedding benchmarks
        container.py      Legacy eval-factory containers
        server.py         HTTP server (Gym protocol)
    benchmarks/           12 built-in benchmarks (all @benchmark + @scorer)
    scoring/
        types.py          ScorerInput dataclass
        text.py           exact_match, fuzzy_match
        pattern.py        multichoice_regex, answer_line, numeric_match
        sandbox.py        code_sandbox (sync) + code_sandbox_async (Sandbox protocol)
        judge.py          LLM-as-judge (needs_judge, judge_score)
        json_schema.py    JSON schema validation
    runner/
        solver.py         Solver protocol (Chat, Completion, VLM, Embedding, Agent, SandboxedAgent)
        nat_solver.py     NatSolver: NAT agent via SSE (/generate/full)
        eval_loop.py      Async parallel eval with back-pressure and sandbox lifecycle
        model_client.py   HTTP client with retry, cache, VLM, embeddings
        deployment.py     Model server lifecycle + DeployConfig
        checkpoint.py     Per-benchmark checkpoint tracking and resume
        regression.py     Run comparison with Mann-Whitney U p-values
        exporters/        WandB, MLflow export plugins
    sandbox/
        base.py           Sandbox protocol, SandboxSpec, ExecResult, OutsideEndpoint
        docker.py         DockerSandbox (async, bridge network, per-task images)
        slurm.py          SlurmSandbox (Pyxis/Enroot, multiplexed slots)
        local.py          LocalSandbox (temp dir, no isolation)
        manager.py        SandboxManager (concurrency, pre-pull, emergency cleanup)
    executors/
        __init__.py       Executor protocol, ProcessState, registry
        local.py          LocalExecutor (in-process or background fork)
        docker.py         DockerExecutor (container run/status/stop)
        slurm.py          SlurmExecutor (sbatch generation + SSH submit)
    eval/
        config.py         EvalConfig Pydantic schema
        local_runner.py   Local execution with service orchestration
        slurm_gen.py      SLURM sbatch generation (colocated + separated)
        containers.py     Container image resolution from containers.toml
    observability/        StepRecord, RuntimeStats, FailureReport
    metrics/              pass@k, bootstrap CI, aggregation
    cli/                  nel eval, serve, validate, list, config, package
    telemetry.py          Opt-in usage analytics
```

## License

Apache-2.0
