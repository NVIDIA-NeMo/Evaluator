# Sandbox Orchestration

Per-problem isolated execution environments for agent evaluations and code execution.

## Overview

NEL provides three levels of isolation:

| Level         | What                                | Implementation                      |
| ------------- | ----------------------------------- | ----------------------------------- |
| **Cluster**   | Where the overall eval runs         | `executors/` — Local, Docker, SLURM |
| **Benchmark** | Per-benchmark environment container | SLURM `node_pools` topology         |
| **Problem**   | Per-problem isolated sandbox        | `sandbox/` — this module            |

The per-problem sandbox is essential for:

- **Agent evaluations** (SWE-bench, OpenHands, terminal-bench): each problem gets an isolated container with its own filesystem, network, and process space.
- **Code execution** (HumanEval, NeMo Skills math): untrusted code runs in a sandbox with resource limits.
- **Multi-turn tool use**: agents in sandboxes need to reach the model server while the orchestrator reaches the agent.

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │          Orchestrator (eval loop)         │
                    │                                          │
                    │  for problem in dataset:                 │
                    │    seed = env.seed(idx)                  │
                    │    sandbox = manager.acquire(spec)       │
                    │    result = solver.solve(task, sandbox)  │
                    │    verify = env.verify(resp, exp, sandbox)│
                    │    manager.release(sandbox)              │
                    └───────────┬──────────────┬───────────────┘
                                │              │
                    ┌───────────▼──┐    ┌──────▼──────────────┐
                    │ Model Server │    │ Sandbox (1/problem)  │
                    │ (vLLM/NIM)   │    │ image: per-task      │
                    │              │◄───┤ bridge network       │
                    │ :8000        │    │ ┌───────────────────┐│
                    └──────────────┘    │ │ Agent or code     ││
                                        │ └───────────────────┘│
                                        └──────────────────────┘
```

## Design Principles

### Sandbox is infrastructure-only

The sandbox knows nothing about agents, solvers, or evaluation. Its job:

- Start a container from a given image
- Provide `exec()`, `upload()`, `download()`
- Expose outside endpoints (model URL) inside the container
- Stop and clean up

Whether we exec a Python script or start a multi-turn agent is the **solver's** decision.

### Async protocol

All sandbox operations are I/O-bound. The eval loop is async. The protocol is fully async to avoid blocking the event loop when running concurrent sandboxes.

### Bridge network by default

With `--network host`, concurrent sandboxes sharing port 3000 collide. Bridge network gives each container its own IP. The orchestrator reaches the container via `container_ip`.

### Verify needs the sandbox

For agent benchmarks, verification (apply patch, run tests) happens inside the same sandbox the agent modified. The sandbox stays alive through: `seed → solve(sandbox) → verify(sandbox) → release`.

## The Per-Task Image Problem

SWE-bench is the critical test case. Every problem has a different Docker image:

- `instance_id = "django__django-11099"` → `swebench/sweb.eval.x86_64.django_1776_django-11099:latest`
- `instance_id = "scikit-learn__scikit-learn-13779"` → `swebench/sweb.eval.x86_64.scikit-learn_1776_scikit-learn-13779:latest`

NEL handles this via three mechanisms (in priority order):

1. **`SandboxSpec` in `SeedResult`**: The environment returns per-problem sandbox requirements directly.
2. **`image_template`**: Config-level template interpolated from metadata: `"swebench/sweb.eval.x86_64.{instance_id}:latest"`.
3. **`default_image`**: Fallback for homogeneous benchmarks.

## Sandbox Backends

### DockerSandbox

Per-problem Docker container on the orchestrator machine.

- Bridge network by default (each container gets its own IP)
- Async operations via `asyncio.create_subprocess_exec`
- `resolve_outside_endpoint()` rewrites `localhost` → `host.docker.internal`

### SlurmSandbox

Per-problem container on SLURM nodes via Pyxis/Enroot.

- **Multiplexed**: multiple sandboxes per node via `--container-name`
- Shared filesystem for fast file transfer when available
- SLURM nodes share the job's network (no URL rewriting needed)

### ECSFargateSandbox

Per-problem container on AWS ECS Fargate. No local Docker required.

- **SSH sidecar**: Each ECS task includes an SSH sidecar container for `exec()`, `upload()`, `download()` via SSH tunnels
- **Bidirectional tunnels**: Forward tunnel (orchestrator → container exec server) and reverse tunnels (container → host proxy/model endpoint)
- **CodeBuild image building**: Per-task Docker images built via AWS CodeBuild and pushed to ECR with content-hash tags for caching
- **`OutsideEndpoint`**: Host-side services (LiteLLM proxy, model server) are made reachable inside the container via reverse SSH tunnels

```yaml
benchmarks:
  - name: harbor://swebench-verified@1.0
    solver:
      type: harbor
      service: model
      agent: openhands
    sandbox:
      type: ecs_fargate
      cluster: harbor-cluster
      subnets: [subnet-abc123]
      security_groups: [sg-def456]
      ecr_repository: 123456789.dkr.ecr.us-west-2.amazonaws.com/harbor-swebench
      ssh_sidecar:
        sshd_port: 2222
        public_key_secret_arn: arn:aws:secretsmanager:...
        private_key_secret_arn: arn:aws:secretsmanager:...
```

### ApptainerSandbox

Per-problem Apptainer/Singularity container for HPC environments where Docker is unavailable. Supports SLURM integration via `salloc`/`srun`.

### LocalSandbox

Async subprocess in a temp directory. No container isolation. For development and testing.

## Configuration

```yaml
benchmarks:
  - name: gym://swebench-server
    sandbox:
      type: docker
      image_template: "swebench/sweb.eval.x86_64.{instance_id}:latest"
      concurrency: 8
      memory: 4g
      network: bridge
```

### SWE-bench on SLURM

```yaml
benchmarks:
  - name: gym://swebench-server
    sandbox:
      type: slurm
      image_template: "swebench/sweb.eval.x86_64.{instance_id}:latest"
      concurrency: 16
      node_pool: sandbox
      slots_per_node: 4

cluster:
  type: slurm
  node_pools:
    sandbox:
      partition: batch
      nodes: 4
```

4 sandbox nodes × 4 slots/node = 16 concurrent sandboxes on 4 nodes.

### Code execution (HumanEval)

```yaml
benchmarks:
  - name: humaneval
    sandbox:
      type: docker
      image: python:3.12-slim
      network: none
      memory: 256m
      cpus: 1
      timeout: 30
      concurrency: 16
```

## SandboxManager

The `SandboxManager` provides:

- **Concurrency control**: semaphore-based limiting of concurrent sandboxes
- **Pre-pull**: bulk image pulling before eval starts (via `env.sandbox_specs()`)
- **Emergency cleanup**: `atexit` and signal handlers to stop leaked containers
- **Spec resolution**: resolves `SandboxSpec` from `SeedResult`, `image_template`, or default image
- **SLURM multiplexing**: round-robin slot allocation across nodes

## SandboxedAgentSolver

The `SandboxedAgentSolver` runs agents inside sandboxes. It supports two modes:

- **exec-server**: no entrypoint → solver execs the agent command via `sandbox.exec()`
- **agent-server**: entrypoint starts an HTTP agent → solver talks to it via `container_ip`

The solver infers the mode from `sandbox.spec.entrypoint`.

## SWE-bench Integration Example

SWE-bench is not a built-in environment. Here's how to integrate it using the sandbox system:

```python
from nemo_evaluator import EvalEnvironment, SeedResult, VerifyResult, benchmark
from nemo_evaluator.sandbox import SandboxSpec


class SWEBenchEnvironment(EvalEnvironment):
    name = "swebench"

    def __init__(self, dataset_path: str):
        super().__init__()
        import json
        with open(dataset_path) as f:
            self._dataset = [json.loads(line) for line in f]

    def _instance_image(self, instance_id: str) -> str:
        safe = instance_id.replace("/", "_").replace("__", "_1776_")
        return f"swebench/sweb.eval.x86_64.{safe}:latest"

    async def sandbox_specs(self) -> list[SandboxSpec]:
        return [
            SandboxSpec(
                image=self._instance_image(row["instance_id"]),
                workdir="/testbed",
            )
            for row in self._dataset
        ]

    async def seed(self, idx: int) -> SeedResult:
        row = self._dataset[idx]
        return SeedResult(
            prompt=row["problem_statement"],
            expected_answer=row["patch"],
            metadata={"instance_id": row["instance_id"]},
            sandbox_spec=SandboxSpec(
                image=self._instance_image(row["instance_id"]),
                workdir="/testbed",
            ),
        )

    async def verify(self, response, expected, sandbox=None, **meta):
        if sandbox is None:
            raise RuntimeError("SWE-bench requires a sandbox for verification")
        await sandbox.exec(f"cd /testbed && git apply /tmp/patch.diff")
        result = await sandbox.exec(
            "cd /testbed && python -m pytest tests/ -x",
            timeout_sec=300,
        )
        passed = result.return_code == 0
        return VerifyResult(
            reward=1.0 if passed else 0.0,
            scoring_details={"method": "swebench_test", "passed": passed},
        )
```

## Adding a New Backend

1. Create `sandbox/my_backend.py` implementing the `Sandbox` protocol.
2. Add a branch in `SandboxManager._create()`.
3. Add a `CustomSandbox` variant or extend the `SandboxConfig` union.
4. Handle emergency cleanup in `SandboxManager._sync_cleanup()`.
