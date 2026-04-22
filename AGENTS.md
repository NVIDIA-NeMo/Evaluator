# NeMo Evaluator -- Agent Contribution Workflow

This file defines how AI coding agents (Claude Code, Cursor, etc.) should contribute features and fixes to this repository. Follow this process for every non-trivial change.

## Repository context

NeMo Evaluator is a unified LLM evaluation framework. The core loop is: `seed -> solve -> verify -> score -> decide`. The evaluator **owns the loop** -- all scoring and aggregation happens inside NEL, never delegated to external servers.

Key directories:

```
src/nemo_evaluator/
  environments/       # EvalEnvironment implementations (byob, gym, harbor, lm_eval, ...)
  solvers/            # Solver implementations (chat, sandbox, nat, openclaw, ...)
  runner/             # Eval loop, artifacts, step logging
  eval/               # Config schema, local runner, SLURM sbatch generator, deployment
  sandbox/            # Sandbox lifecycle, manager, Docker/SLURM backends
  serving/            # FastAPI HTTP server (nel serve)
  metrics/            # Aggregation, confidence intervals, pass@k
  scoring/            # Scoring primitives (fuzzy_match, numeric_match, judge, ...)
  cli/                # Click CLI (nel eval run, nel serve, nel compare, ...)
tests/                # Offline + network test suites
examples/             # Getting started guide, example configs, integration scripts
docs/                 # Architecture, tutorials, API reference, deployment guides
```

## RFC-first development

Before implementing a feature, check for an RFC or design document that describes the change. If one exists, treat it as the source of truth for requirements and architecture decisions.

When no RFC exists and the change is significant (new environment type, new solver, architectural change), write a design summary in your plan phase and get it reviewed before proceeding.

## Contribution workflow

Every non-trivial change follows a five-phase process. Do not skip phases.

### Phase 1: Plan

Before writing any code, produce a written plan that covers:

- **Problem statement**: What is broken or missing, in one paragraph.
- **Proposed solution**: How you intend to fix or build it. Reference specific files and functions.
- **Files to change**: List every file you expect to modify or create.
- **Open questions**: Anything ambiguous that needs human input.

Keep the plan concise. Do not over-engineer simple tasks (< 3 files, single concern).

### Phase 2: Architect review

Adopt the persona of a **grumpy principal architect** and review your own plan. The architect's job is to find structural problems:

- Does this break the "evaluator owns the loop" principle?
- Does it introduce unnecessary coupling between modules?
- Is there class/abstraction proliferation where a simple function would do?
- Are there simpler alternatives that achieve 80% of the value with 20% of the complexity?
- Does the naming make sense to someone reading the code for the first time?
- Will this be painful to maintain in 6 months?

Write the architect's feedback as a short review (bullet points). If the architect rejects the plan, revise it before proceeding.

### Phase 3: Senior engineer review

Adopt the persona of a **pragmatic senior engineer** and review the architect's feedback against the plan. The senior engineer's job is to:

- Validate that the architect's concerns are real, not theoretical.
- Propose concrete implementation adjustments (not just "this is wrong" but "do X instead").
- Flag anything the architect missed: edge cases, test coverage, migration paths.
- Confirm the final file list and change scope.

Produce a short review with actionable items. After this phase, the plan is locked.

### Phase 4: Implement

Write the code. Follow these rules:

- **No narration comments.** Do not add comments that just describe what the code does. Comments should explain non-obvious intent, trade-offs, or constraints.
- **Minimal changes.** Only touch files in the agreed plan. If you discover something else needs changing, note it but do not expand scope silently.
- **Preserve style.** Match the existing code style: type hints, docstring format, import ordering, naming conventions.
- **No fallback shims.** When making breaking changes, break cleanly. Do not add deprecation wrappers unless explicitly asked.
- **Test as you go.** If you add a new function, add a test for it. If you change behavior, update existing tests.

### Phase 5: Pre-commit checks

Before committing, always:

1. **Lint**: Run linter checks on all modified files. Fix any errors you introduced. Do not fix pre-existing lints unless they are in code you modified.
2. **Test**: Run `python -m pytest tests/ -v --tb=short -m "not network" -x`. All tests must pass. If a test fails because of your change, fix it. If a test fails for an unrelated reason, note it but do not silently skip it.
3. **Verify imports**: If you moved or renamed a module, grep for the old import path across the entire repo. Update all references.
4. **CHANGELOG**: For user-visible changes, add an entry under the current version in `CHANGELOG.md`.

Only after all checks pass, create the commit.

## Commit messages

Use clear, imperative commit messages. Focus on "why" not "what".

Format: `<type>: <short imperative description>`. Optionally include an issue reference in brackets if the change maps to a tracked issue.

```
feat: add server_cmd support for native Gym servers on SLURM

The sbatch generator now honors ServiceConfig.server_cmd for gym
services instead of hardcoding `nel serve`. This enables running
any server that speaks /seed_session + /verify as a managed service.
```

Prefix with `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:` as appropriate.

## Design principles

These are non-negotiable. If your change violates any of them, redesign.

1. **Evaluator owns the loop.** Seed, solve, verify, score, and decide all happen inside NEL. Never delegate scoring or aggregation to an external server. The verify endpoint returns a reward; NEL computes all metrics from those rewards.

2. **Environments are interchangeable.** Any `EvalEnvironment` works with any `Solver` (where the combination makes sense). Do not create environment-solver coupling.

3. **Solvers are stateless.** A solver receives a `SeedResult` and returns a `SolveResult`. It does not hold per-problem state. Connection state (HTTP clients, etc.) is fine.

4. **Config is the contract.** All user-facing behavior is controlled through `EvalConfig` YAML. If it can't be expressed in config, it probably shouldn't be a feature.

5. **Sandbox lifecycle is explicit.** `NoSandbox`, `StatefulSandbox`, `StatelessSandbox` are the three modes. The eval loop picks one based on the seed result. Do not add implicit sandbox behavior.

6. **Abstract over server internals.** When integrating external systems (Gym, Harbor, lm-eval), NEL should not import or depend on their internal code. Communicate via HTTP, CLI, or file I/O. The `server_cmd` pattern (start an opaque process, health-check it, connect) is preferred over tight coupling.

## Testing conventions

- **Offline tests** (`tests/`): No network access, no real model calls. Use `FixturedEnvironment`, `CachedSolver`, `MockSandbox` from `tests/conftest.py`.
- **Network tests** (`@pytest.mark.network`): Allowed to download HuggingFace datasets. Run separately.
- **Golden fixtures** (`tests/fixtures/`): Generated by `tests/generate_fixtures.py` against a real API. Stored as JSON. Used by `CachedSolver` for deterministic offline replay.
- **No SLURM tests.** SLURM execution is tested by sbatch generation (unit test the generated script string), not by actually submitting jobs.
- **Prefer parametrized tests.** When testing the same function with multiple inputs, use `@pytest.mark.parametrize` with descriptive `ids` instead of separate test methods. Import shared fixtures and helpers at module level, not inside each test.

## Deployment and evaluation runs

When asked to run an evaluation on SLURM:

1. **Create a config YAML** in `examples/configs/` following the existing patterns. Every config should declare:
   - `services`: model server (vllm/sglang/api) and any benchmark servers (gym with `server_cmd`)
   - `benchmarks`: one or more benchmark URIs with model reference, concurrency, and problem count
   - `cluster`: `type: slurm`, `partition`, `account` (PPP), `nodes`, `gres`, `walltime`, `container_image`
   - `output`: directory and report formats

2. **Container images** are auto-selected from `containers.toml` based on the benchmark URI scheme (`gym://` -> gym variant, `lm-eval://` -> lm-eval variant, etc.). Override with `cluster.container_image` when using a specific pre-built image.

3. **Dry-run first** to inspect the generated sbatch script:
   ```
   nel eval run <config>.yaml --dry-run
   ```

4. **Submit** from the appropriate login node:
   ```
   nel eval run <config>.yaml --submit
   ```

5. **Model deployment on SLURM**: Use `type: vllm` or `type: sglang` services. The sbatch generator starts the model server as a background process, waits for health, then runs the eval. Set `tensor_parallel_size` to match the number of GPUs requested. For large models (>70B), allow a generous `startup_timeout` (900s+).

6. **Gym servers on SLURM**: Use `type: gym` with `server_cmd`. The gym container (`Dockerfile.gym`) has all Gym resource servers pre-installed at `/opt/Gym`. No mounts or extra setup needed.

## What NOT to do

- Do not add dependencies without checking `pyproject.toml` extras. Core must stay lightweight.
- Do not create new top-level directories without discussion.
- Do not add `print()` statements. Use `logging.getLogger(__name__)`.
- Do not hardcode paths, API keys, or model names.
- Do not skip the plan phase "because the change is obvious." If it touches more than 2 files, plan it.
- Do not push to `main` directly.
