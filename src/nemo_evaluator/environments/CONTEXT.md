# Environment Context

Domain language for environment adapters and Bring Your Own Benchmark support.

## Language

**BYOB Benchmark**:
A user-defined **Benchmark** declared with decorators instead of a handwritten **Environment** subclass.
_Avoid_: BYOB, BYOB Agent, custom environment

**Benchmark Definition**:
The registered declaration of a **BYOB Benchmark**, including its dataset source, seed construction rules, and verification function.
_Avoid_: Benchmark Config, eval config, benchmark run config

**Dataset Spec**:
The source declaration in a **Benchmark Definition** that tells a **BYOB Benchmark** how to load raw problem rows.
_Avoid_: Dataset, data, dataset rows

**Dataset Row**:
One loaded record from a **Dataset Spec**, before the BYOB environment turns it into a **Seed**.
_Avoid_: Problem, sample, task

**Row Preparation**:
The deterministic transformation of a **Dataset Row** before seed construction.
_Avoid_: Preprocessing, seeding, verification

**Prompt Template**:
The format string in a **Benchmark Definition** used to construct a prompt from a **Dataset Row**.
_Avoid_: Seed, prompt, instruction

**Custom Seed Function**:
A BYOB hook that directly constructs the common **Seed** for a **Dataset Row**, bypassing prompt-template construction.
_Avoid_: Row Preparation, scorer, prompt template

**BYOB Scorer**:
The benchmark-specific verification function attached to a **Benchmark Definition**.
_Avoid_: Scoring, metric, judge

**Scorer Input**:
The data package passed to a **BYOB Scorer**, containing the solution response, target, row metadata, benchmark extra config, and optional sandbox.
_Avoid_: Seed, sample, scoring config

**Target Field**:
The field name in a **Dataset Row** whose value becomes the **Target** used for verification.
_Avoid_: Target, expected answer

**Target**:
The value read from the **Target Field** and passed to the **BYOB Scorer**.
_Avoid_: Target Field, expected answer field

**Image Build Request**:
A BYOB declaration that a benchmark-specific container image must be built before evaluation starts.
_Avoid_: Sandbox Spec, sandbox image, build config

## Relationships

- A **Benchmark Definition** declares exactly one **BYOB Benchmark**.
- A **Benchmark Definition** has exactly one **Dataset Spec**.
- A **Dataset Spec** may point to HuggingFace data, a local data file, or a callable that returns rows.
- A **Dataset Spec** loads zero or more **Dataset Rows**.
- **Row Preparation** may transform each **Dataset Row** before seeding.
- A **Dataset Row** becomes a **Seed** during seeding.
- A **Benchmark Definition** may use a **Prompt Template** to construct a **Seed**.
- A **Benchmark Definition** may use a **Custom Seed Function** to construct a **Seed**.
- A **Custom Seed Function** takes precedence over a **Prompt Template**.
- A **Benchmark Definition** has zero or one **BYOB Scorer**.
- A **BYOB Scorer** receives exactly one **Scorer Input** per verification.
- A **BYOB Scorer** participates in **Verification**, not aggregate **Scoring**.
- A **Benchmark Definition** names a **Target Field**.
- A **Target Field** identifies the **Target** within a **Dataset Row**.
- A **Target** becomes the common expected answer used during **Verification**.
- A **Benchmark Definition** may declare an **Image Build Request**.
- An **Image Build Request** prepares images before attempts acquire **Sandboxes**.
- A **BYOB Benchmark** is presented to the evaluation loop through a generated **Environment**.
- A **BYOB Benchmark** extends the common **Benchmark** vocabulary from the root context.

## Example dialogue

> **Dev:** "Is BYOB here the benchmark decorator API or the directory-based agent solver?"
> **Domain expert:** "Use **BYOB Benchmark** for the decorator API. Say **BYOB Agent** only for the agent solver."

## Flagged ambiguities

- "BYOB" was used for both Bring Your Own Benchmark and Bring Your Own Agent; resolved: use **BYOB Benchmark** for the `@benchmark` + `@scorer` API in this context.
- "benchmark definition" and "benchmark config" were easy to confuse; resolved: **Benchmark Definition** is the decorator-captured declaration, while Benchmark Config remains runtime/YAML configuration.
- "dataset" was used for both the source declaration and loaded data; resolved: **Dataset Spec** is the source declaration.
- "row" and "problem" were easy to conflate; resolved: **Dataset Row** is the loaded record before seeding, while **Problem** is the evaluation-loop item.
- "prepare_row" was easy to confuse with seeding; resolved: **Row Preparation** transforms rows before seed construction.
- "prompt" was used for both a template and the constructed seed field; resolved: **Prompt Template** is the row-formatting template, while **Seed** is the evaluation-loop input package.
- "scorer" was easy to confuse with aggregate scoring; resolved: **BYOB Scorer** is benchmark-specific verification logic, while common **Scoring** is evaluator-owned metric computation.
- "target" was used for both a field name and a value; resolved: **Target Field** is the row field name, while **Target** is the value passed to the scorer.
- "image build request" and "sandbox spec" were easy to conflate; resolved: **Image Build Request** prepares images ahead of time, while Sandbox Spec describes per-attempt runtime needs.
