# Common Evaluation Context

Shared domain language for NeMo Evaluator across benchmarks, solvers, services, sandboxes, scoring, reporting, and orchestration. This context exists so engineering work uses one vocabulary for the evaluation loop and adjacent systems.

## Language

**Benchmark**:
A reusable evaluation task source that can seed problems and verify answers.
_Avoid_: Dataset, test suite, harness

**Environment**:
The executable adapter that presents a **Benchmark** to the evaluation loop through seeding and verification.
_Avoid_: Benchmark, harness, runner

**Benchmark Run**:
One configured execution of a **Benchmark** inside an **Evaluation Run**, including solver, repeats, scoring, sandbox, and concurrency.
_Avoid_: Benchmark, job

**Evaluation Run**:
One invocation of NeMo Evaluator that executes one or more **Benchmark Runs** and produces artifacts, metrics, and reports.
_Avoid_: Run, job, experiment

**Problem**:
One benchmark item evaluated once per repeat, from seeding through verification.
_Avoid_: Sample, task, item

**Attempt**:
One solve-and-verify pass for a **Problem** within a **Benchmark Run**.
_Avoid_: Repeat, sample, trial

**Seed**:
The environment-produced problem package for an **Attempt**, including the prompt or messages, expected answer, metadata, and any sandbox requirements.
_Avoid_: Prompt, input, task

**Solver**:
The strategy that turns a **Seed** into a **Solution**.
_Avoid_: Model, service, agent

**Solution**:
The solver-produced output for an **Attempt**, including response text and any trajectory or solver-side metadata.
_Avoid_: Answer, response, completion

**Service**:
A configured external or managed runtime endpoint that a **Solver**, judge, verifier, or resource integration can call.
_Avoid_: Solver, model, server

**Execution Backend**:
The runtime target that launches and manages an **Evaluation Run**, such as local, Docker, or SLURM.
_Avoid_: Environment, cluster, runner

**Cluster Config**:
The part of an eval config that selects and configures the **Execution Backend**.
_Avoid_: Executor config, deployment config

**Node Pool**:
A named group of compute resources within a SLURM **Cluster Config** that services or sandboxes can reference.
_Avoid_: Partition, node group

**Sandbox**:
A per-problem isolated execution environment used by an **Attempt** for agent execution, code execution, or verification.
_Avoid_: Service, benchmark container, model server

**Verification**:
The environment-owned check that compares a **Solution** against a **Seed** and returns a reward plus evidence.
_Avoid_: Scoring, judging, grading

**Scoring**:
The evaluator-owned process that turns verification outputs, judge metrics, and configured scoring rules into benchmark metrics.
_Avoid_: Verification, aggregation

**Reward**:
The numeric outcome of **Verification** for one **Attempt**.
_Avoid_: Score, metric

**Score**:
An aggregate or derived metric over rewards and scoring details for a **Benchmark Run** or **Evaluation Run**.
_Avoid_: Reward

**Artifact**:
Any persisted output of an **Evaluation Run** or **Benchmark Run**, including step logs, bundles, reports, exports, traces, and checkpoints.
_Avoid_: Result, output file

**Artifact Access**:
The default expectation that non-credential **Artifacts** from an **Evaluation Run** are readable and traversable, but not writable, by other cluster users.
_Avoid_: Output permissions, result permissions

**Credential File**:
A launch-time operational file that carries secret material and must stay private even when **Artifacts** are shared.
_Avoid_: Artifact, output file

**Comparison**:
Paired analysis of two **Evaluation Runs** or **Benchmark Runs** to identify score deltas, behavioral flips, and statistical evidence of regression.
_Avoid_: Gate, evaluation

**Baseline**:
The reference run in a **Comparison**.
_Avoid_: Control, old model

**Candidate**:
The run being evaluated against the **Baseline** in a **Comparison**.
_Avoid_: Experiment, new model

**Quality Gate**:
A policy evaluation over one **Baseline** and one **Candidate** across one or more **Benchmark Runs**, producing a release verdict.
_Avoid_: Comparison, regression test

**Gate Policy**:
The configured thresholds, required benchmarks, tiers, and metric directions used by a **Quality Gate**.
_Avoid_: Config, rule file

**Verdict**:
The top-level GO, NO-GO, or INCONCLUSIVE decision from a **Quality Gate**.
_Avoid_: Status, result

## Relationships

- An **Evaluation Run** contains one or more **Benchmark Runs**.
- A **Benchmark Run** executes exactly one **Benchmark**.
- A **Benchmark** is presented to the evaluation loop by exactly one **Environment** during a **Benchmark Run**.
- A **Benchmark** contains zero or more **Problems**.
- A **Problem** belongs to exactly one **Benchmark**.
- A **Problem** has one or more **Attempts** in a **Benchmark Run**.
- An **Attempt** evaluates exactly one **Problem**.
- An **Attempt** starts with exactly one **Seed**.
- A **Seed** is produced by an **Environment**.
- A **Solver** consumes a **Seed**.
- An **Attempt** produces exactly one **Solution** unless it fails before solving.
- A **Solver** may call one or more **Services**.
- An **Attempt** may use a **Sandbox**.
- A **Sandbox** is infrastructure for an **Attempt**, not a solver or scoring owner.
- An **Evaluation Run** is launched by exactly one **Execution Backend**.
- A **Cluster Config** selects and configures one **Execution Backend**.
- A **Node Pool** belongs to a SLURM **Cluster Config**.
- An **Environment** performs **Verification** for a **Solution**.
- **Verification** returns exactly one **Reward** for an **Attempt**.
- **Scoring** consumes verification outputs and configured scoring rules.
- **Scoring** produces one or more **Scores**.
- An **Evaluation Run** produces one or more **Artifacts**.
- A **Benchmark Run** may produce one or more **Artifacts**.
- Non-credential **Artifacts** should have **Artifact Access** by default.
- **Artifact Access** applies regardless of the **Execution Backend** that produced the **Artifacts**.
- **Artifact Access** applies only to **Artifacts** produced or owned by the current **Evaluation Run**.
- **Artifact Access** lets other cluster users inspect **Artifacts**, not mutate an **Evaluation Run**.
- Failure to apply **Artifact Access** should not change the success or failure of an **Evaluation Run**.
- Logs are **Artifacts** and should have **Artifact Access** by default.
- Sharded **Evaluation Runs** should apply **Artifact Access** incrementally as shard **Artifacts** are produced, not only after final merge.
- Non-credential operational metadata is an **Artifact** and should have **Artifact Access** by default.
- A **Credential File** is not an **Artifact** even if it is stored near **Artifacts**.
- **Artifact Access** must not make a **Credential File** readable by other cluster users.
- A **Comparison** has exactly one **Baseline** and exactly one **Candidate**.
- A **Baseline** and **Candidate** should evaluate the same **Problems** for valid paired analysis.
- A **Quality Gate** evaluates one **Baseline** and one **Candidate** using a **Gate Policy**.
- A **Quality Gate** produces exactly one **Verdict**.
- A **Comparison** may provide diagnostic evidence for a **Quality Gate**, but does not decide the **Verdict**.
- Per-module context files may extend this shared vocabulary for local concerns.
- Per-module context files should not redefine common terms without calling out the conflict.

## Example dialogue

> **Dev:** "For each repeat, does the **Solver** get the **Problem** directly?"
> **Domain expert:** "No. Each **Attempt** starts from a **Seed** produced by the **Environment**. The **Solver** turns that Seed into a **Solution**, then the Environment performs **Verification** and returns a **Reward**."

## Flagged ambiguities

- "benchmark" was used for both the reusable task source and one configured execution; resolved: **Benchmark** is the task source, **Benchmark Run** is the configured execution.
- "environment" was used interchangeably with "benchmark"; resolved: **Benchmark** is the user-facing task source, **Environment** is the executable adapter.
- "environment" was also used for deployment targets; resolved: use **Execution Backend** for local, Docker, SLURM, and similar runtime targets.
- "sample" and "problem" were both used for benchmark inputs; resolved: **Problem** is the input item, while "sample" is reserved for metrics and reporting when needed.
- "repeat" is a configuration count, not the domain noun; resolved: one repeated solve-and-verify pass is an **Attempt**.
- "prompt" was used for the full input package; resolved: **Seed** is the package, while prompt/messages are fields inside it.
- "answer" was used for solver output; resolved: **Solution** is the full solver output, while answer-specific terms are reserved for scoring.
- "model" was used to mean solver strategy and service endpoint; resolved: **Solver** is the strategy, **Service** is the configured runtime endpoint.
- "sandbox" was used for several isolated runtime scopes; resolved: **Sandbox** is per-problem infrastructure for an **Attempt**, not a benchmark container or model service.
- "verification" and "scoring" were used interchangeably; resolved: **Verification** is the environment-owned check, while **Scoring** is evaluator-owned metric computation.
- "reward" and "score" were used interchangeably; resolved: **Reward** is per-attempt, while **Score** is aggregate or derived.
- "result" was used as a broad persisted-output term; resolved: **Artifact** is the broad persisted-output term, while result remains local to code return objects and report schemas.
- "output dirs" was used to include both run artifacts and credential-bearing sidecars; resolved: **Artifact Access** applies to non-credential **Artifacts**, while **Credential Files** stay private.
- "output dir" was used as a permission boundary; resolved: the boundary is **Artifacts** produced or owned by the current **Evaluation Run**, not every filesystem entry under a configured directory.
- "compare" and "gate" were used as generic release checks; resolved: **Comparison** is diagnostic paired analysis, while gate terminology is reserved for policy decisions.
- "status", "result", and "verdict" were all used for gate outputs; resolved: **Verdict** is the top-level release decision from a **Quality Gate**.
