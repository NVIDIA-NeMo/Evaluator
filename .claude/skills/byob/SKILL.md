---
name: byob-benchmark
description: Use when adding, wrapping, or onboarding a benchmark into nemo-evaluator (not NeMo-Gym training) — covers BYOB `@benchmark`/`@scorer` authoring, agentic task wrappers, choosing between BYOB vs an existing `gym://`/`harbor://`/`skills://`/`lm-eval://` URI vs a direct `EvalEnvironment` subclass, and the scorer-input/sandbox/verifier traps. Trigger even when the user just says "add a benchmark", "wrap MMLU", "write a scorer", or "integrate eval X" without mentioning BYOB.
---

# BYOB benchmark authoring

Use this skill when adding or onboarding a benchmark to nemo-evaluator. Keep it
as a decision guide, not an API reference. For exact signatures and examples,
inspect the local code and `docs/tutorials/byob.md`.

## Non-negotiables

- **NEL owns seed -> solve -> verify -> score -> aggregate.** Never call an
  upstream harness command that runs the agent loop or aggregates scores.
- **Local code wins.** If old docs, upstream examples, or copied benchmarks
  disagree with this repo, follow this repo and mention the drift.
- **Config is the user contract.** Prefer existing URI/config/playbook paths
  over new Python code when they already express the benchmark.

## Source-of-truth pass

Before editing, inspect only the pieces relevant to the selected path:

- `docs/tutorials/byob.md` for the human-facing BYOB workflow.
- `src/nemo_evaluator/environments/custom.py` for BYOB decorator behavior.
- `src/nemo_evaluator/environments/base.py` for seed and verify contracts.
- `src/nemo_evaluator/solvers/` before relying on solver metadata, workspace
  sync, sandbox access, tools, transcripts, or artifacts.
- `src/nemo_evaluator/benchmarks/pinchbench.py` and `terminal_bench_v1.py` as
  canonical agentic BYOB shapes (workspace per task, task-id cache, mixed
  solver compatibility).

## Pick the onboarding path

| Benchmark shape | Preferred path |
|---|---|
| One row becomes one prompt and one target | Simple BYOB benchmark |
| Rows need deterministic cleanup or shuffling | BYOB with `prepare_row` |
| Prompt needs messages, metadata, or sandbox specs | BYOB with `seed_fn` |
| Tasks need workspaces, assets, verifier scripts, or solver artifacts | Agentic BYOB wrapper |
| Tasks are already in a supported task-suite format | Existing URI/playbook, or a thin registered wrapper |
| Lifecycle cannot fit BYOB cleanly | Direct `EvalEnvironment` |

Do not make a new BYOB module when a config-only `gym://`, `harbor://`,
`skills://`, or `lm-eval://` path already covers the benchmark.

## Simple BYOB benchmark

Use this for text-only, extraction, multiple-choice, or numeric-answer
benchmarks. Canonical references in this repo: `mmlu.py` and `gpqa.py`
(multiple choice with `prepare_row` + `multichoice_regex`), `gsm8k.py`
(numeric answer with `numeric_match`), `math500.py` (extracted answer line).

Essential work:

- Add one benchmark module with a BYOB benchmark plus scorer.
- Register it by adding a single import line to
  `src/nemo_evaluator/benchmarks/__init__.py` — the `@benchmark` decorator
  registers the environment at module load.
- Add offline scorer coverage; add row-preparation coverage when
  `prepare_row` is non-trivial.
- Smoke with `nel validate -b <name> --samples 10` once the module loads.

Critical constraints:

- Scorers receive `ScorerInput`, not raw strings. Use response, target,
  metadata, config, and sandbox fields exposed by the local type.
- The built-in scoring primitives (`exact_match`, `multichoice_regex`,
  `numeric_match`, `answer_line`, ...) already return a dict in the right
  shape — just `return multichoice_regex(sample)` and friends.
- If you build the dict yourself, the eval loop reads `correct` first, then
  `reward`. Extra keys become scoring details.
- Use `seed_fn` when the expected answer is structured or when prompt
  construction is not a plain row-template operation.

## Agentic BYOB wrapper

Use this when BYOB still fits, but the task needs workspace files, manifests,
solver artifacts, deterministic verifiers, or background services.

Essential contract:

- Dataset rows can be the full parsed task dicts. When the verifier code
  (e.g. an embedded `grade()` function) ships with the task package, a
  module-global cache keyed by task id is **required**, not optional — the
  eval loop builds `ScorerInput` from response, expected answer, and verify
  metadata only, so the scorer cannot reach solver internals or
  non-serializable task fields any other way. See `pinchbench.py` for the
  canonical shape.
- `seed_fn` should create isolated per-problem state, materialize assets,
  return prompts/messages, and include verifier metadata such as task id,
  workspace handles, taxonomy, or runtime values.
- The configured NEL solver still owns solving. Inspect that solver before
  claiming support for workspace sync, tools, transcripts, or sandbox access.
- The scorer reloads task metadata and verifies solver output, workspace state,
  sandbox state, or documented artifacts. It returns per-attempt reward data;
  NEL aggregates.
- Clean up background services and temporary workspaces with
  `atexit.register(...)` at module scope. The `@benchmark` decorator does not
  expose a `close()` hook on the generated environment, so process-level
  cleanup is the practical pattern (see `pinchbench.py`).

Critical traps:

- Keep `seed_fn` idempotent. NEL may call it during sandbox pre-pull, and one
  `SeedResult` is reused across repeats for the same problem.
- A host `workspace_path` only helps if the selected solver runs locally or
  explicitly syncs that path. For generic sandboxed tasks, use sandbox specs
  and verify through `sample.sandbox`.
- `SolveResult.trajectory` is not automatically passed to `ScorerInput` —
  `ScorerInput` carries only response, target, metadata, config, and sandbox.
  Normalize any solver-internal artifact you need into a documented response,
  metadata, workspace, or sandbox path before verification.
- Avoid solver/server modes that set `SolveResult.reward` when you expect BYOB
  scoring. If reward is already set, the eval loop skips `env.verify()`.
- Start benchmark-owned services from `seed_fn` only when startup is idempotent
  and the runtime can expose reachable URLs to the solver sandbox. Otherwise,
  use a direct environment with lifecycle hooks.

## Image and sandbox lifecycle

- Use `@image_builder` for benchmark-owned images that must be provisioned
  before attempts start.
- Use `SeedResult.sandbox_spec` for the agent sandbox image, files, env,
  volumes, and workdir.
- Use `SeedResult.verify_sandbox_spec` when solve and verify must run in
  separate containers. Pair it with capture/apply commands when state must move
  between containers.
- Do not build images, clone large repos, or install dependencies from a
  scorer — scorers run per attempt and per repeat, so once-per-eval work
  belongs in `@image_builder` or the environment's `prepare()` hook.

## Pre-packaged task suite

Use this when tasks already ship in a local or remotely resolvable task-suite
format, or when the repo already exposes them through a URI scheme, registry,
or playbook. Canonical reference for the "thin registered subclass that
prepends an `ImageBuildRequest`" pattern: `nmp_harbor.py` (subclasses
`HarborEnvironment`, overrides `image_build_requests()` to bake a base image
before any per-task image).

Essential work:

- Prefer the existing environment/URI/playbook path over BYOB decorators.
- Add a thin registered subclass only for missing lifecycle behavior, such as
  preparing a non-native release layout, adding benchmark-owned image builds,
  or enforcing benchmark-specific verification/readiness behavior. The
  override lives on the parent class hook (`async def image_build_requests`),
  not on the `@image_builder` decorator — that decorator is for BYOB
  benchmarks the module itself defines.
- Keep the verification boundary NEL-owned: task packages provide prompts,
  sandbox metadata, and verifier commands; NEL runs the solver, records
  per-attempt rewards, and aggregates metrics.

Tests should focus on config/playbook parsing, mocked URI or registry
resolution, lifecycle behavior, sandbox specs, image builds, timeouts, and
readiness behavior touched by the change.

## Direct environment

Use a direct `EvalEnvironment` subclass when BYOB would hide important
lifecycle or state:

- long-lived connections or services need `prepare()`/`close()`;
- seeding or verification cannot be represented by `seed_fn`, sandbox specs,
  and a scorer;
- BYOB would require fragile global state.

Even with a direct environment, preserve the NEL loop. The environment seeds
and verifies; NEL solves, records rewards, and aggregates.

## Review checklist

- Did we choose the simplest path: config/URI, simple BYOB, agentic BYOB,
  pre-packaged wrapper, or direct environment?
- Does the implementation keep solving, verification recording, and aggregation
  inside NEL?
- Are workspace, sandbox, artifact, and service assumptions proven against the
  selected solver/runtime?
- Are benchmark-specific tests focused on the behavior we changed rather than
  generic repo mechanics?
