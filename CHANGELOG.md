# Changelog

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

- Deleted duplicate scoring modules (math_equal, extraction, mcq, exact_match)
- Environments own their scoring via `@scorer` functions
- Kept judge.py and json_schema.py as NEL-native capabilities
- Scoring primitives: `multichoice_regex`, `answer_line`, `numeric_match`, `fuzzy_match`, `exact_match`, `code_sandbox`, `needs_judge`

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
