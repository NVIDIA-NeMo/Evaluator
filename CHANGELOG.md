# Changelog

## 0.5.0 (2026-03-07)

### CLI Redesign

- **`nel eval` command group**: `run`, `status`, `stop`, `report` subcommands replace `nel run`, `nel slurm eval`, `nel harness run`, and `nel container-eval`.
- **`nel list`**: Unified benchmark discovery across built-in, NeMo Skills, and lm-eval sources. Replaces `nel list-environments`, `nel list-harnesses`, `nel list-skills`.
- **`nel serve`**: Flat command for serving benchmarks as HTTP endpoints.
- Removed all legacy commands: `nel run`, `nel slurm`, `nel harness`, `nel container-eval`, `nel list-environments`, `nel list-harnesses`, `nel list-skills`.

### Unified Config Schema

- **Two-tier `EvalConfig`**: Simple mode (`model:` + `benchmarks:`) for quick evaluations, advanced mode (`services:` + `benchmarks:`) for multi-model setups with managed infrastructure.
- **`ClusterConfig`**: SLURM, Docker, and local execution from a single config file.
- **`OutputConfig`**: Multi-format report generation (HTML, Markdown, CSV, JSON, LaTeX).
- Removed old `evaluation: { tasks: [] }` config format and `config_schema.py`.

### Evaluation Orchestration

- **`eval/local_runner.py`**: Service lifecycle management (vLLM, SGLang, Gym servers) with health polling.
- **`eval/slurm_gen.py`**: Self-contained sbatch script generation from `EvalConfig`.
- **`eval/ssh.py`**: Remote SLURM submission via SSH.
- **`eval/lifecycle.py`**: Executor-aware process management (`status`/`stop`) for local, SLURM, and Docker.

### Bug Fixes

- Fixed `str.format()` crash in sbatch template generation.
- Fixed PID file written before output directory exists.
- Fixed Gym service template generating nonexistent CLI subcommand.
- Fixed shell injection via unescaped quotes in system prompts.
- Fixed `num_examples` silently dropped for all environment types.
- Renamed `models.EvalConfig` → `RunSnapshot` to avoid collision with `eval.config.EvalConfig`.

### Cleanup

- Deleted dead modules: `cli/run.py`, `cli/slurm.py`, `cli/harness.py`, `config_schema.py`, `environments/definitions.py`.
- Removed all backward-compatibility aliases and deprecation stubs.
- Updated all documentation and examples to use new CLI and config format.

## 0.4.0 (2026-03-07)

### Architecture

- **Everything is an Environment**: Collapsed adapters, harnesses, and environments into one `EvalEnvironment` base class. Gym, Skills, PI, lm-eval -- all are environments.
- **`@benchmark` + `@scorer` API**: Universal benchmark definition mechanism. All 11 built-in benchmarks use it. External BYOB uses the same API.
- **Solver protocol**: `ChatSolver`, `CompletionSolver`, `AgentSolver`. Eval loop calls `solver.solve()` instead of hardcoded `client.chat()`. Unblocks agent benchmarks.
- **Executor layer**: `LocalExecutor`, `DockerExecutor`, `SlurmExecutor` manage full lifecycle (deploy model, run eval, collect results, tear down).
- **Model deployment**: `NIMDeployment`, `VLLMDeployment`, `ProcessModelDeployment`, `APIDeployment` with health polling and graceful shutdown.
- **Judge post-processing**: Eval loop optionally runs LLM-as-judge scoring after `verify()`.

### Benchmarks (11 built-in)

MMLU, MMLU-Pro, MATH-500, GPQA Diamond, GSM8K, DROP, MGSM, TriviaQA, HumanEval (Docker sandbox), SimpleQA (judge), HealthBench (judge).

### Environments

- `GymEnvironment` + `ManagedGymEnvironment` (server lifecycle)
- `SkillsEnvironment` (auto-prepares datasets)
- `LMEvalEnvironment` (generate_until tasks)
- `PIEnvironment` (Prime Intellect verifiers)

### Scoring

- Scoring package with dedicated modules: `text.py`, `pattern.py`, `sandbox.py`, `judge.py`, `json_schema.py`
- Benchmarks own their scoring via `@scorer` functions calling reusable primitives
- Primitives: `exact_match`, `fuzzy_match`, `multichoice_regex`, `answer_line`, `numeric_match`, `code_sandbox`, `needs_judge`
- LLM-as-judge: `JudgeScoringConfig`, `judge_score`, `needs_judge` signal
- JSON schema validation: `extract_json`, `validate_json_schema`

### Eval Loop

- Async parallel with semaphore-bounded concurrency (default 32)
- Back-pressure pipeline: seeds 2x buffer, never loads full dataset into task queue
- Proper resource cleanup: `env.close()` and `solver.close()` in finally blocks

### Security

- Docker sandbox: code piped via stdin (not `-c`), `--pids-limit`, `--read-only`, `--security-opt no-new-privileges`
- API keys passed via env vars (not CLI args) in AgentSolver
- Prompt template KeyError warnings (not silent swallows)

### Tests

- Integration tests for `run_evaluation()` end-to-end
- Golden tests for all scoring primitives
- Scorer import tests for each built-in benchmark
- 144 total tests

## 0.3.0 (2026-02-25)

Initial architecture with unified eval loop, harness adapters, and observability.
