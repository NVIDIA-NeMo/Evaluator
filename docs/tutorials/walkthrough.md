# Interactive Walkthrough

A guided tour of NeMo Evaluator's main features, explained through real config files. Each section builds on the previous one — by the end you will understand services, solvers, scoring, sandboxes, suites, cluster deployment, and sharding.

## Prerequisites

```bash
pip install -e ".[scoring]"
export NVIDIA_API_KEY="your-api-key-here"
```

All examples below reference configs in `examples/configs/`. You can run any of them directly:

```bash
nel eval run examples/configs/01_gsm8k_chat.yaml
```

---

## 1. The Simplest Evaluation

**Config:** `01_gsm8k_chat.yaml`

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: gsm8k
    max_problems: 5
    solver:
      type: simple
      service: nemotron
```

Every config has two required sections:

**`services`** declares the model endpoints you want to evaluate. Each service has:

- **`type`** — `api` means an external endpoint you connect to (not one NEL deploys).
- **`url`** — The full endpoint URL, including the path (`/v1/chat/completions`).
- **`protocol`** — How to format requests (`chat_completions`, `completions`, or `responses`). This is separate from the URL because experimental endpoints may use a different path than the standard one.
- **`model`** — The model identifier sent in the API request.
- **`api_key`** — Supports `${ENV_VAR}` expansion so you never hardcode secrets.

**`benchmarks`** is a list of evaluations to run. Each benchmark has:

- **`name`** — A built-in benchmark name (`gsm8k`), a URI (`skills://gpqa`, `lm-eval://mmlu`), or a remote server (`gym://host:port`).
- **`solver`** — Tells NEL *how* to interact with the model. `type: simple` sends the prompt and parses the response. `service: nemotron` points to the service declared above.

The GSM8K benchmark has a built-in scorer that extracts the last number from the model's response and compares it to the ground truth. No extra scoring config is needed.

```bash
nel eval run examples/configs/01_gsm8k_chat.yaml
```

:::{admonition} What happens under the hood
:class: dropdown

1. NEL loads the GSM8K dataset (1,319 math problems).
2. `max_problems: 5` limits the run to the first 5.
3. For each problem, the `simple` solver sends the prompt to the `nemotron` service via `/v1/chat/completions`.
4. The benchmark's built-in scorer extracts the answer and compares to ground truth.
5. Results are written to `./eval_results/` with pass@k scores, trajectories, and runtime stats.
:::

---

## 2. Completions Protocol (Base Models)

**Config:** `02_mmlu_completions.yaml`

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/completions
    protocol: completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}
    generation:
      max_tokens: 64

benchmarks:
  - name: mmlu
    max_problems: 5
    solver:
      type: simple
      service: nemotron
    scoring:
      metrics:
        - type: scorer
          name: mcq_extract
```

Two new concepts here:

**`generation`** controls LLM generation parameters — `max_tokens`, `temperature`, `top_p`, etc. For multiple-choice, we only need a short answer, so `max_tokens: 64` keeps things fast.

**`scoring.metrics`** overrides the benchmark's default scorer. Here we use `mcq_extract` which extracts a letter (A-D) from the response. Each metric has a `type` and `name` — the `scorer` type runs a built-in scoring function.

Notice the URL and protocol changed — `/v1/completions` with `protocol: completions`. The `simple` solver automatically adapts: for `chat_completions` it sends a messages array, for `completions` it sends a raw text prompt.

---

## 3. External Benchmark Sources

**Config:** `03_mmlu_lmeval.yaml` and `04_mmlu_pro_skills.yaml`

NEL isn't limited to its 15 built-in benchmarks. URI schemes plug in external benchmark libraries:

```yaml
benchmarks:
  # lm-evaluation-harness (pip install nemo-evaluator[lm-eval])
  - name: lm-eval://mmlu_generative
    solver:
      type: simple
      service: nemotron

  # NeMo Skills (pip install nemo-evaluator[skills])
  - name: skills://mmlu-pro
    solver:
      type: simple
      service: nemotron
```

| URI Scheme | Source | Install |
|------------|--------|---------|
| `lm-eval://task` | lm-evaluation-harness | `pip install -e ".[lm-eval]"` |
| `skills://name` | NeMo Skills | `pip install -e ".[skills]"` |
| `vlmevalkit://dataset` | VLMEvalKit | VLMEvalKit on `PYTHONPATH` |
| `gym://host:port` | Remote Gym server | (none) |
| `container://image#task` | Legacy container | Docker |

The service and solver config is identical regardless of the benchmark source — you are always in control of which model gets called and how.

---

## 4. LLM-as-Judge (Two Services)

**Config:** `05_simpleqa_judge.yaml`

```yaml
services:
  solver:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

  judge:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: simpleqa
    max_problems: 5
    solver:
      type: simple
      service: solver
    scoring:
      include_defaults: false
      metrics:
        - type: judge
          name: correctness
          service: judge
```

This config declares **two separate services**: one for solving and one for judging. In production you would use a stronger model as the judge.

The scoring section introduces:

- **`include_defaults: false`** — Disables the benchmark's built-in scorer. Only the explicitly listed metrics run.
- **`type: judge`** — An LLM-as-judge metric. The `service` field points to the judge model, and NEL automatically constructs a judge prompt from the response and expected answer.

:::{admonition} Self-judge prevention
:class: warning
NEL validates that the judge service is different from the solver's service. If you accidentally point both to the same service, NEL raises an error — unless you explicitly set `allow_self_judge: true` on the metric.
:::

---

## 5. Code Execution in Docker Sandboxes

**Config:** `06_humaneval_docker.yaml`

```yaml
benchmarks:
  - name: humaneval
    max_problems: 5
    solver:
      type: simple
      service: nemotron
    sandbox:
      type: docker
      image: python:3.12-slim
      timeout: 60.0
```

HumanEval generates Python code. To score it, NEL executes the generated code inside a Docker container:

- **`sandbox.type: docker`** — Each problem gets its own isolated container.
- **`sandbox.image`** — The Docker image to use. HumanEval needs Python, so `python:3.12-slim`.
- **`sandbox.timeout`** — Kill the container after 60 seconds (prevents infinite loops).

The sandbox is *infrastructure-only* — it knows nothing about scoring. The HumanEval benchmark's scorer extracts code from the model response, writes it into the container alongside test cases, and checks the exit code.

Other sandbox options: `memory`, `cpus`, `network` (bridge/host/none), `concurrency` (max parallel containers).

---

## 6. Agentic Evaluation with Harbor

**Config:** `07_swebench_harbor.yaml`

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}
    interceptors:
      - name: turn_counter
        config:
          max_turns: 100

benchmarks:
  - name: harbor://swebench-verified@1.0
    max_problems: 3
    solver:
      type: harbor
      service: nemotron
      agent: openhands-sdk
      agent_kwargs:
        max_iterations: 100
    sandbox:
      type: ecs_fargate
      cluster: harbor-cluster
      subnets: [subnet-abc123]
      # ... (ECS configuration)
```

This is a significant step up. Three new concepts:

**`interceptors`** are LiteLLM proxy callbacks that intercept every request between the agent and the model. Here, `turn_counter` limits the agent to 100 turns. Other interceptors can log requests, modify payloads, or cache responses. Each service gets its own set of interceptors.

**`solver.type: harbor`** runs a full autonomous agent (OpenHands, Terminus-2, etc.) instead of a single model call. The `agent` field selects which agent framework, and `agent_kwargs` passes configuration to it.

**`sandbox.type: ecs_fargate`** runs each problem in an AWS ECS Fargate container — no local Docker required. Each SWE-bench problem gets its own container with the correct codebase pre-installed. An SSH sidecar enables bidirectional tunnels between the orchestrator and the container.

:::{admonition} Harbor benchmark URIs
:class: tip
`harbor://swebench-verified@1.0` downloads task definitions from the Harbor registry via sparse git checkout. Each task includes a Dockerfile, test script, and metadata. Tasks are cached locally after first download.
:::

---

## 7. Gym Integration

**Config:** `09_gym_runtime.yaml`

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

  swe_gym:
    type: gym
    url: http://localhost:11413

benchmarks:
  - name: swebench-multilingual
    solver:
      type: gym_delegation
      service: nemotron
      gym_service: swe_gym
      gym_agent: openhands
      trust_reward: false
    sandbox:
      type: docker
      timeout: 1800.0
```

Here we see a **`type: gym`** service — a connection to a running NeMo Gym server. The `gym_delegation` solver delegates the agent execution to the Gym server while NEL handles verification independently.

Key fields:

- **`gym_service`** — Points to the `swe_gym` service (the Gym server).
- **`trust_reward: false`** — NEL verifies results independently rather than trusting Gym's reward. Set to `true` for Gym-trusted scoring.

---

## 8. NEL-Driven Tool Calling (ReActSolver)

**Configs:** `21_tool_calling_gym.yaml`, `22_tool_calling_sandbox.yaml`, `23_tool_calling_combined.yaml`

The `gym_delegation` and `harbor` solvers delegate the agentic loop to an external framework (Gym Agent Server, Harbor agent). This is convenient but **opaque** — NEL only sees the final response, not the individual model calls, tool calls, and conversation turns.

The `tool_calling` solver changes this: **NEL drives the loop itself**. It calls the model, parses tool calls, dispatches them, and records every turn — giving full observability.

### Gym HTTP Tools

```yaml
services:
  model:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/llama-3.3-nemotron-super-49b-v1
    api_key: ${NVIDIA_API_KEY}
  tools:
    type: gym
    url: http://localhost:8000

benchmarks:
  - name: swebench-multilingual
    solver:
      type: tool_calling
      service: model
      resource_service: tools
      max_turns: 50
```

Here, `resource_service: tools` points to a Gym Resource Server. NEL discovers available tools via `/openapi.json` and dispatches tool calls via `POST /{tool_name}`. Every model call, tool result, and conversation turn is captured in the ATIF trajectory.

### Sandbox Tools (NEL as the Agent)

```yaml
benchmarks:
  - name: swebench-verified
    solver:
      type: tool_calling
      service: model
      sandbox_tools: true
      max_turns: 100
      response_mode: sandbox_artifact
      system_prompt: |
        You are a software engineer fixing a bug in a Python repository.
        You have these tools: bash, str_replace_editor, file_read, file_write.
        Use bash to explore the repo and run tests. Use str_replace_editor to
        make targeted edits. When done, the sandbox state is captured automatically.
    sandbox:
      type: docker
      timeout: 1800.0
```

With `sandbox_tools: true`, NEL provides a canonical set of tools that execute inside the Docker sandbox:

| Tool | What it does |
|------|-------------|
| `bash` | Execute shell commands (`sandbox.exec()`) |
| `file_read` | Read file contents (`sandbox.download()`) |
| `file_write` | Write file contents (`sandbox.upload()`) |
| `str_replace_editor` | View, create, or make targeted replacements in files |

File operations use `sandbox.upload()` / `sandbox.download()` internally — no content is ever embedded in shell commands.

Key fields:

- **`sandbox_tools: true`** — enables the four sandbox tools listed above. Requires a sandbox config.
- **`response_mode: sandbox_artifact`** — the "answer" is the sandbox state (e.g. a git patch), not the last model message. Use for SWE-bench.
- **`max_turns`** — maximum model call rounds before stopping. The solver sets `error: "max_turns_exhausted"` if hit.
- **`tool_timeout`** — per-tool-call timeout in seconds (default: 180).
- **`max_output_chars`** — truncation limit for tool output fed back to the model (default: 16384). Large outputs are truncated with head+tail preservation.

### Combined (HTTP + Sandbox)

```yaml
benchmarks:
  - name: research-bench
    solver:
      type: tool_calling
      service: model
      resource_service: tools       # web_search, find_in_page from Gym
      sandbox_tools: true            # bash, file_read, file_write
      max_turns: 30
    sandbox:
      type: docker
```

Both tool sources can be combined: the model sees tools from the Gym Resource Server *and* sandbox tools. Tool calls are routed to the correct backend by name.

### Observability Gains

| Aspect | Delegated (`harbor`/`gym_delegation`) | NEL-driven (`tool_calling`) |
|--------|---------------------------------------|---------------------------|
| Model calls | Opaque (inside agent framework) | Every call: tokens, latency, content |
| Tool calls | Opaque | Every tool: name, args, result, latency |
| Conversation | Final response only | Full multi-turn transcript |
| Trajectory | Reconstructed post-hoc | Native ATIF per-turn |
| Failure attribution | "Agent failed" | "bash() failed on turn 3 with exit code 1" |

---

## 9. Vision-Language Models

**Config:** `13_vlmevalkit_mmbench.yaml`

```yaml
services:
  vila:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/vila-3b
    api_key: ${NVIDIA_API_KEY}

benchmarks:
  - name: vlmevalkit://MMBench_DEV_EN
    max_problems: 5
    solver:
      type: simple
      service: vila
      image_detail: high
```

The `simple` solver handles VLM benchmarks automatically when the dataset includes images. The `image_detail` field (`auto`, `low`, `high`) controls the resolution of images sent to the model. VLMEvalKit datasets provide image data alongside text prompts.

---

## 10. Multi-Benchmark Suites

**Config:** `17_suite_release.yaml`

```yaml
services:
  nemotron:
    type: api
    url: https://integrate.api.nvidia.com/v1/chat/completions
    protocol: chat_completions
    model: nvidia/nemotron-3-super-120b-a12b
    api_key: ${NVIDIA_API_KEY}

sandboxes:
  humaneval_code:
    type: docker
    image: python:3.12-slim
    timeout: 60.0

benchmarks:
  - name: gsm8k
    max_problems: 5
    solver:
      type: simple
      service: nemotron

  - name: mmlu
    max_problems: 5
    solver:
      type: simple
      service: nemotron

  - name: humaneval
    max_problems: 5
    solver:
      type: simple
      service: nemotron
    sandbox: humaneval_code

  - name: simpleqa
    max_problems: 5
    solver:
      type: simple
      service: nemotron
```

Two important features here:

**`sandboxes`** is a top-level dict of named sandbox configurations. Instead of inlining the sandbox config in every benchmark, you define it once and reference it by name: `sandbox: humaneval_code`. This avoids duplication when multiple benchmarks share the same sandbox setup.

**Multiple benchmarks** in a single config run sequentially. Each gets its own output directory. If one fails, the others still run (failure isolation). You can resume a partially completed suite:

```bash
nel eval run examples/configs/17_suite_release.yaml --resume
```

---

## 11. Auto-Deployed vLLM on SLURM

**Config:** `15_slurm_gsm8k_vllm.yaml`

```yaml
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
    max_problems: 50
    solver:
      type: simple
      service: model

cluster:
  type: slurm
  walltime: "02:00:00"
  node_pools:
    compute:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"
```

This is where NEL becomes a "docker-compose for evals":

**`type: vllm`** — NEL *deploys* the model server for you. It starts a vLLM process, waits for health, runs the benchmark, then tears it down. Same for `sglang` and `trt_llm`.

**`cluster`** declares infrastructure. Instead of running locally, this generates an sbatch script and submits it to SLURM.

**`node_pools`** defines named groups of resources. Here, `compute` is 1 node with 4 GPUs. Services reference pools via `node_pool: compute`.

**`tensor_parallel_size: 4`** tells vLLM to shard the model across 4 GPUs — matching the `gres: "gpu:4"` in the node pool. NEL validates this consistency.

```bash
nel eval run examples/configs/15_slurm_gsm8k_vllm.yaml
```

This generates `eval.sbatch`, submits it, and monitors progress.

---

## 12. Heterogeneous SLURM Jobs

**Config:** `16_slurm_swebench_harbor.yaml`

```yaml
services:
  model:
    type: vllm
    model: nvidia/Llama-3.1-70B-Instruct
    protocol: chat_completions
    tensor_parallel_size: 4
    port: 8000
    node_pool: gpu

sandboxes:
  swebench:
    type: slurm
    node_pool: sandbox
    concurrency: 4
    slots_per_node: 4
    timeout: 1800.0

benchmarks:
  - name: swebench-verified
    max_problems: 10
    solver:
      type: harbor
      service: model
      agent: openhands
    sandbox: swebench

cluster:
  type: slurm
  walltime: "04:00:00"
  node_pools:
    gpu:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"
    sandbox:
      partition: batch
      nodes: 2
```

This uses **two node pools**: `gpu` (1 node with 4 GPUs for vLLM) and `sandbox` (2 CPU nodes for agent sandboxes). NEL generates a SLURM heterogeneous job that allocates both pools simultaneously.

The `swebench` sandbox uses `type: slurm` — containers run via Pyxis/Enroot on the sandbox nodes. With `slots_per_node: 4` across 2 nodes, there are 8 concurrent sandbox slots.

---

## 13. Sharded Evaluation

**Config:** `15a_slurm_gsm8k_vllm_sharded.yaml`

```yaml
cluster:
  type: slurm
  walltime: "02:00:00"
  shards: 4
  node_pools:
    compute:
      partition: batch
      nodes: 1
      ntasks_per_node: 1
      gres: "gpu:4"
```

Adding `shards: 4` to the cluster config turns the SLURM job into an **array job**. NEL generates `#SBATCH --array=0-3`, and each array task evaluates a disjoint slice of the dataset.

After all tasks complete, merge the results:

```bash
nel eval merge ./eval_results
```

This discovers `shard_0/`, `shard_1/`, etc., deduplicates any overlapping results, and produces a single merged bundle with recomputed pass@k and confidence intervals.

:::{admonition} Sharding without SLURM
:class: tip
You can shard on any infrastructure using environment variables:

```bash
NEL_SHARD_IDX=0 NEL_TOTAL_SHARDS=4 nel eval run config.yaml -o ./results/shard_0
NEL_SHARD_IDX=1 NEL_TOTAL_SHARDS=4 nel eval run config.yaml -o ./results/shard_1
# ... after all complete:
nel eval merge ./results
```
:::

---

## 14. Legacy Container Harnesses

**Config:** `14_container_nemo_skills.yaml`

```yaml
benchmarks:
  - name: "container://registry.example.com/nemo-skills:26.03#gsm8k"
    solver:
      type: container
      service: nemotron
      uri: "container://registry.example.com/nemo-skills:26.03#gsm8k"
    sandbox:
      type: docker
      timeout: 3600.0
```

The `container://` scheme runs an existing eval harness container as an opaque box. The container owns the full eval loop — NEL parses `results.yml` and `eval_factory_metrics.json` from the output.

This is the least observable mode (no per-request trajectories), but it supports any existing harness without modification.

---

## Config Anatomy Cheat Sheet

Every NeMo Evaluator config follows this structure:

```yaml
# 1. Services — what endpoints you are evaluating
services:
  <name>:
    type: api | vllm | sglang | trt_llm | gym | nat_agent
    url: <full endpoint URL>
    protocol: chat_completions | completions | responses
    model: <model identifier>
    # Optional:
    api_key: ${ENV_VAR}
    generation: { temperature, top_p, max_tokens, seed, stop, frequency_penalty, presence_penalty }
    interceptors: [{ name, config }, ...]
    proxy_verbose: false
    depends_on: [<other service names>]

# 2. Sandboxes (optional) — reusable sandbox configs
sandboxes:
  <name>:
    type: docker | slurm | ecs_fargate | apptainer | local | none
    # type-specific fields...

# 3. Benchmarks — what to evaluate
benchmarks:
  - name: <benchmark name or URI>
    solver:
      type: simple | harbor | tool_calling | gym_delegation | openclaw | container
      service: <service name>
      # solver-specific fields...
    # Optional:
    sandbox: <sandbox name or inline config>
    scoring:
      metrics: [{ type, name, service }, ...]
    max_problems: <int>
    repeats: <int>
    max_concurrent: <int>
    timeout: <float>

# 4. Cluster (optional) — where to run
cluster:
  type: local | slurm | docker
  # cluster-specific fields...

# 5. Output (optional) — reporting and export
output:
  dir: ./eval_results
  export: [inspect, wandb, mlflow]
```

---

## What's Next?

| Goal | Next step |
|------|-----------|
| Write a custom benchmark | {doc}`byob` |
| Integrate with NeMo Gym | {doc}`gym-integration` |
| Compare runs and detect regressions | {doc}`compare` |
| Set up multi-benchmark quality gates | {doc}`quality-gate` |
| Scale to large datasets | {doc}`distributed-eval` |
| Understand the architecture | {doc}`../architecture/index` |
| Browse all 25 configs | [`examples/configs/`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/examples/configs) |
