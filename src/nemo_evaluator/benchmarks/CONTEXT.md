# Built-in Benchmark Context

Domain language for first-party benchmarks under `src/nemo_evaluator/benchmarks/`, especially benchmark-specific task, workspace, and grading concepts that sit above generic BYOB environment machinery.

## Language

**Built-in Benchmark**:
A first-party **Benchmark** that ships with NeMo Evaluator and is registered by the repository at import time.
_Avoid_: BYOB Benchmark, external benchmark, benchmark config

**PinchBench Task**:
An upstream markdown-authored benchmark unit containing frontmatter, prompt, expected behavior, grading criteria, and optional workspace files or grading code.
_Avoid_: Problem, sample, dataset row

**Task Workspace**:
The per-task filesystem workspace materialized from a **PinchBench Task** before solving, then inspected during verification for agent-created files.
_Avoid_: Sandbox, working directory, artifact directory

**Workspace Evidence**:
The agent-created or modified files in a **Task Workspace** that PinchBench verification may inspect directly or include in the judge prompt.
_Avoid_: Artifact, output files, workspace dump

**PinchBench Transcript**:
The solver interaction transcript, in the upstream PinchBench message-envelope shape, that PinchBench graders use as verification evidence.
_Avoid_: Trajectory, ATIF trajectory, step log

**Automated PinchBench Grader**:
Task-authored grading code attached to a **PinchBench Task** that computes reward from the **PinchBench Transcript** and **Task Workspace** without calling a judge service.
_Avoid_: Scorer, BYOB Scorer, grade function

**PinchBench Rubric**:
The natural-language grading criteria from a **PinchBench Task** used to construct the judge prompt for judge-scored verification.
_Avoid_: Expected answer, judge prompt, target

**PinchBench Grading Mode**:
The task-declared choice of verification path for a **PinchBench Task**: automated, judge-scored, or hybrid.
_Avoid_: Grading type, benchmark type, scorer type

**Judge-Scored PinchBench Task**:
A **PinchBench Task** whose reward is produced by a judge service using the **PinchBench Rubric**, **PinchBench Transcript**, and any **Workspace Evidence**.
_Avoid_: Needs-judge task, LLM judge task, judge benchmark

**Hybrid PinchBench Task**:
A **PinchBench Task** whose reward combines an **Automated PinchBench Grader** result with a judge-produced rubric score.
_Avoid_: Fallback task, mixed scorer, combined scorer

**PinchBench Judge Prompt**:
The rendered prompt sent to the judge service for a **Judge-Scored PinchBench Task** or **Hybrid PinchBench Task**, assembled from the task prompt, expected behavior, **PinchBench Rubric**, **PinchBench Transcript**, and **Workspace Evidence**.
_Avoid_: Rubric, expected answer, judge input

## Relationships

- A **Built-in Benchmark** is a **Benchmark** in the common evaluation context.
- A **Built-in Benchmark** may use the BYOB `@benchmark` and `@scorer` machinery without being a user-defined **BYOB Benchmark**.
- Benchmark-specific task, workspace, and grading language belongs here instead of in the generic environment context.
- A **PinchBench Task** is upstream-authored source material for the PinchBench **Built-in Benchmark**.
- A **PinchBench Task** becomes a common **Problem** only when NeMo Evaluator seeds it for a **Benchmark Run**.
- A **PinchBench Task** may materialize one **Task Workspace** for an **Attempt**.
- A **Task Workspace** may contain initial task files and agent-created files.
- A **Task Workspace** provides verification evidence; it is not the **Sandbox** that executes the solver.
- **Workspace Evidence** is read from a **Task Workspace** during verification.
- **Workspace Evidence** may affect **Reward** for automated, judge-scored, and hybrid PinchBench tasks.
- A **PinchBench Transcript** may be read from a **Task Workspace** during verification.
- A **PinchBench Transcript** is verification evidence, not an observability artifact.
- An **Automated PinchBench Grader** belongs to one **PinchBench Task**.
- An **Automated PinchBench Grader** participates in **Verification** for one **Attempt**.
- An **Automated PinchBench Grader** does not own aggregate **Scoring**.
- A **PinchBench Task** may declare one **PinchBench Rubric**.
- A **PinchBench Rubric** guides judge-scored verification, not extractive answer matching.
- A **PinchBench Task** has exactly one **PinchBench Grading Mode**.
- **PinchBench Grading Mode** selects the verification path for an **Attempt**.
- A **Judge-Scored PinchBench Task** has judge-scored **PinchBench Grading Mode**.
- A **Judge-Scored PinchBench Task** requires a judge **Service** during verification.
- A **Hybrid PinchBench Task** has hybrid **PinchBench Grading Mode**.
- A **Hybrid PinchBench Task** requires both an **Automated PinchBench Grader** and a judge **Service** for full verification.
- A **Hybrid PinchBench Task** may fall back to the automated reward if the judge service cannot produce a rubric score.
- A **PinchBench Judge Prompt** is constructed during verification for each judge-scored or hybrid **Attempt**.
- A judge **Service** consumes a **PinchBench Judge Prompt** and returns evidence used to compute **Reward**.

## Example dialogue

> **Dev:** "Can the **Sandbox** just be inspected directly by the PinchBench scorer?"
> **Domain expert:** "No. The solver may use a Sandbox, but PinchBench verification reads the **Task Workspace**. The **Task Workspace** contains **Workspace Evidence** and may contain the **PinchBench Transcript** used by the **Automated PinchBench Grader** or the **PinchBench Judge Prompt**."

## Flagged ambiguities

- Built-in benchmarks and user-defined BYOB benchmarks can share the same decorator machinery; resolved: **Built-in Benchmark** is first-party benchmark content shipped by this repository, while **BYOB Benchmark** remains the user-defined decorator API term in the environment context.
- "task" is avoided for generic evaluation-loop items, but PinchBench's upstream source units are task markdown files; resolved: use **PinchBench Task** for upstream task source material and **Problem** for the evaluation-loop item.
- "workspace" was easy to confuse with sandbox runtime infrastructure; resolved: **Task Workspace** is the per-task file corpus and verification evidence, while **Sandbox** is attempt execution infrastructure.
- "artifact" is already the persisted output of an evaluation run; resolved: **Workspace Evidence** is benchmark evidence used during verification, not a run artifact.
- "trajectory" is overloaded between solver metadata, ATIF exports, and scoring evidence; resolved: **PinchBench Transcript** is the upstream-shaped transcript consumed by PinchBench graders.
- "scorer" is already used for BYOB verification functions and aggregate scoring; resolved: **Automated PinchBench Grader** names task-authored grading code inside PinchBench.
- "expected answer" implies extractive answer matching; resolved: **PinchBench Rubric** names behavioral grading criteria for judge-scored PinchBench verification.
- `grading_type` is the implementation field name; resolved: use **PinchBench Grading Mode** in domain docs for the task's verification-path choice.
- `needs_judge` is an implementation signal; resolved: use **Judge-Scored PinchBench Task** for the domain concept.
- "fallback task" overemphasizes the error path; resolved: **Hybrid PinchBench Task** names the normal two-signal verification path.
- "rubric" and "judge prompt" were easy to conflate; resolved: **PinchBench Rubric** is task-authored criteria, while **PinchBench Judge Prompt** is the rendered prompt sent to a judge service.
