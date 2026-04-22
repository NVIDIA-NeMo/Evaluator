(template-home)=

# NeMo Evaluator Documentation

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

````{div} sd-d-flex-row
```{button-ref} get-started/install
:ref-type: doc
:color: primary
:class: sd-rounded-pill sd-mr-3

Install
```

```{button-ref} get-started/quickstart
:ref-type: doc
:color: secondary
:class: sd-rounded-pill sd-mr-3

Quickstart
```
````

---

## Get Started

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`download;1.5em;sd-mr-1` Installation
:link: get-started/install
:link-type: doc
Install from source and configure extras (scoring, skills, harbor, proxy).
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Quickstart
:link: get-started/quickstart
:link-type: doc
Run your first evaluation in under 5 minutes.
:::
::::

## Features

- **Everything is an Environment.** Built-in benchmarks, NeMo Skills, Gym remotes, lm-eval tasks, and VLMEvalKit datasets all resolve through one registry.
- **`@benchmark` + `@scorer`.** Define a complete benchmark in under 10 lines of Python.
- **Pluggable solvers.** `simple`, `harbor`, `tool_calling`, `gym_delegation`, `openclaw` — swap inference strategy per benchmark via config.
- **Cluster backends.** Run locally, in Docker, or on SLURM clusters with automatic model deployment.
- **Resilient suites.** Per-benchmark checkpointing with failure isolation. Resume partially completed suites with `--resume`.
- **Statistical regression.** Compare runs with McNemar's exact test, paired flip analysis, and confidence intervals.  Gate releases across benchmark suites with per-benchmark policy thresholds.
- **15 built-in benchmarks.** MMLU, MMLU-Pro, MATH-500, GPQA, GSM8K, DROP, MGSM, TriviaQA, HumanEval, SimpleQA, HealthBench, PinchBench, XSTest, SWE-bench Verified, SWE-bench Multilingual.

## Tutorials

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`mortar-board;1.5em;sd-mr-1` Interactive Walkthrough
:link: tutorials/walkthrough
:link-type: doc
Guided tour of all main features through real config examples — start here.
:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Write Your Own Benchmark
:link: tutorials/byob
:link-type: doc
Define a complete benchmark with `@benchmark` + `@scorer` in under 10 lines.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Gym Integration
:link: tutorials/gym-integration
:link-type: doc
Serve benchmarks for NeMo Gym training and consume remote Gym environments.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` NeMo Skills Integration
:link: tutorials/skills-integration
:link-type: doc
Use NeMo Skills benchmarks with full per-request observability.
:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Distributed Evaluation
:link: tutorials/distributed-eval
:link-type: doc
Scale to thousands of problems with SLURM, Kubernetes, Ray, or manual sharding.
:::

:::{grid-item-card} {octicon}`git-compare;1.5em;sd-mr-1` Compare Runs
:link: tutorials/compare
:link-type: doc
Diagnose what changed between two runs of the same benchmark with `nel compare`.
:::

:::{grid-item-card} {octicon}`shield-check;1.5em;sd-mr-1` Quality Gates
:link: tutorials/quality-gate
:link-type: doc
Turn benchmark thresholds into a suite-level `GO / NO-GO / INCONCLUSIVE` decision with `nel gate`.
:::
::::

## Architecture & Deployment

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`cpu;1.5em;sd-mr-1` Architecture
:link: architecture/index
:link-type: doc
How the system works: environments, solvers, execution modes, and observability.
:::

:::{grid-item-card} {octicon}`server;1.5em;sd-mr-1` Deployment Guide
:link: deployment/index
:link-type: doc
Deploy on SLURM, Docker, Kubernetes, Ray, and CI/CD pipelines.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Benchmarks
:link: evaluation/benchmarks
:link-type: doc
All 15 built-in benchmarks with scoring details.
:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` API Reference
:link: api/index
:link-type: doc
Python API and CLI reference.
:::
::::

:::{toctree}
:hidden:
Home <self>
:::

:::{toctree}
:caption: Get Started
:hidden:

get-started/install
get-started/quickstart
:::

:::{toctree}
:caption: Tutorials
:hidden:

tutorials/index
tutorials/walkthrough
tutorials/byob
tutorials/gym-integration
tutorials/skills-integration
tutorials/legacy-containers
tutorials/distributed-eval
tutorials/compare
tutorials/quality-gate
tutorials/adapters
:::

:::{toctree}
:caption: Architecture
:hidden:

architecture/index
architecture/sandbox
:::

:::{toctree}
:caption: Evaluation
:hidden:

evaluation/benchmarks
evaluation/scoring
:::

:::{toctree}
:caption: Deployment
:hidden:

deployment/index
deployment/local
deployment/docker
deployment/slurm
deployment/kubernetes
deployment/ray
deployment/ci-regression
:::

:::{toctree}
:caption: Patterns
:hidden:

deployment-patterns
:::

:::{toctree}
:caption: References
:hidden:

api/index
:::
