# Nemotron 3 Ultra 550B A55B — v0.3 evaluation recipes

Native agentic evaluation configs for Nemotron 3 Ultra on `nemo-evaluator` 0.3.x:
SWE-bench (Verified / Multilingual / Pro) and Terminal-bench (Hard / 2.0). Each runs
through `harbor` against an ECS Fargate sandbox, one config per benchmark.

The instruct suite is run on the v0.2 stack — see the
[v0.2 recipes](../v0.2/README.md). For the cross-generation overview and the full
benchmark table, see [`../reproducibility.md`](../reproducibility.md).

---

## Common setup

### 1. Install `nemo-evaluator`

The `harbor` extra pulls the agentic deps:

```bash
python3 -m venv .venv
.venv/bin/pip install "nemo-evaluator[harbor]==0.3.*"
```

### 2. Policy model

Required by every benchmark — the model under test:

| `.env` key | Config block | Set / recommended model |
|---|---|---|
| `NVIDIA_API_KEY` | `services.nemotron` | endpoint `url` + `model` — your model under test |

Default params (set in each config): `temperature 1.0`, `top_p 0.95`, thinking on.
To self-host, point `services.nemotron.url` at your vLLM endpoint; the served model
name must match `services.nemotron.model`. Full serve commands (BF16 / FP8 / NVFP4 —
parsers, TP, env vars) are in the header of
[`../v0.2/local_nemotron-3-ultra-550b-a55b.yaml`](../v0.2/local_nemotron-3-ultra-550b-a55b.yaml).

### 3. Sandbox prerequisites

Every benchmark runs inside an ECS Fargate sandbox via `harbor`, which needs:

| Requirement | Config block | Notes |
|---|---|---|
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | AWS env | ECS Fargate sandbox infra (auto-discovered from AWS SSM for the region) |
| `HARBOR_ECS_REGION` | `sandbox.region` | optional override; per-config defaults below |
| harbor ECR | `sandbox.ecr_repository` | replace `<AWS_ACCOUNT_ID>` with your harbor ECR account |
| Terraform applied | — | for the sandbox region (creates the SSM parameter) |

The ECS Fargate sandbox infrastructure is not provisioned by `nemo-evaluator`. You
must self-provision it in your own AWS account — a reference Terraform stack that
creates the exact topology `harbor`'s `ECSFargateSandbox` backend expects is in
[`terraform/`](../../../../terraform/README.md).

---

## Running

Do the [Common setup](#common-setup) once, then run any config with `nel eval run`:

```bash
nel eval run local_nemotron-3-ultra-550b-a55b_swebench-verified.yaml
```

### Output

Results are written to each config's `output.dir` (e.g. `./results/swebench_verified`).
Override per run with `-o output.dir=/absolute/path`.

---

## Benchmarks

### SWE-bench (Verified / Multilingual / Pro)

Agentic coding: the OpenHands agent solves each task inside an ECS Fargate sandbox;
its patch is graded by the task's own test suite. SWE-bench Pro additionally swaps in
a custom system prompt and the instruction template under [`./prompts/`](./prompts/).
All share the AWS / ECR / Terraform [sandbox prerequisites](#3-sandbox-prerequisites).

| Config | Default region | Repeats | `max_concurrent` |
|---|---|---|---|
| `local_nemotron-3-ultra-550b-a55b_swebench-verified.yaml` | `us-east-2` | 3 | 10 |
| `local_nemotron-3-ultra-550b-a55b_swebench-multilingual.yaml` | `us-east-2` | 3 | 10 |
| `local_nemotron-3-ultra-550b-a55b_swebench-pro.yaml` | `us-east-1` | 3 | 8 |

```bash
nel eval run local_nemotron-3-ultra-550b-a55b_swebench-verified.yaml
nel eval run local_nemotron-3-ultra-550b-a55b_swebench-multilingual.yaml
nel eval run local_nemotron-3-ultra-550b-a55b_swebench-pro.yaml
```

### Terminal-bench (Hard / 2.0)

Agentic terminal: the Terminus-2 agent drives a shell inside a stateful ECS Fargate
sandbox; each task is graded by its own `test.sh`. Same
[sandbox prerequisites](#3-sandbox-prerequisites) as SWE-bench.

| Config | Default region | Repeats | Notes |
|---|---|---|---|
| `local_nemotron-3-ultra-550b-a55b_terminal-bench-hard.yaml` | `us-west-1` | 8 | **keep `us-west-1`** — task images are cached in that region's harbor ECR; running elsewhere forces a full CodeBuild rebuild of every task image (slow, can fail) |
| `local_nemotron-3-ultra-550b-a55b_terminal-bench-2.yaml` | `us-east-1` | 8 | — |

```bash
nel eval run local_nemotron-3-ultra-550b-a55b_terminal-bench-hard.yaml
nel eval run local_nemotron-3-ultra-550b-a55b_terminal-bench-2.yaml
```

---

## Troubleshooting

- **Keys not found** — export them or use `.env`; names must match exactly.
- **Sandbox won't start** — check AWS credentials, that the harbor ECR
  `<AWS_ACCOUNT_ID>` is set, and that Terraform has been applied for the region
  (the SSM parameter must exist).
- **Terminal-bench Hard rebuilds every image** — you're not in `us-west-1`; switch back.

## Expected results

Comparable to the model card; expect slight variation (`temperature > 0` with
repeats consensus). Never report sub-sampled / limited runs.

## License

Apache 2.0 — see the repository `LICENSE`.
