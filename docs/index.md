# NeMo Evaluator

LLM evaluation framework: benchmark environments, pluggable solvers, multi-format reporting.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Get Started
:link: get-started/install
:link-type: doc

Install, configure, and run your first evaluation in 5 minutes.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Build your own benchmark, integrate with Gym, and more.
:::

:::{grid-item-card} Architecture
:link: architecture/index
:link-type: doc

How the system works: environments, solvers, execution modes, and observability.
:::

:::{grid-item-card} Deployment
:link: deployment/index
:link-type: doc

Deploy on SLURM, Docker, and CI/CD pipelines.
:::
::::

## Highlights

- **Everything is an Environment.** Built-in benchmarks, NeMo Skills, Gym remotes, lm-eval tasks, and VLMEvalKit datasets all resolve through one registry.
- **`@benchmark` + `@scorer`.** Define a complete benchmark in under 10 lines of Python.
- **Pluggable solvers.** `simple`, `harbor`, `tool_calling`, `gym_delegation`, `openclaw` -- swap inference strategy per benchmark via config.
- **Executors.** Run locally, in Docker, or on SLURM clusters with automatic model deployment.
- **Resilient suites.** Per-benchmark checkpointing with failure isolation. Resume partially completed suites with `--resume`.
- **Statistical regression.** Compare runs with confidence intervals and Mann-Whitney U p-values.
- **15 built-in benchmarks.** MMLU, MMLU-Pro, MATH-500, GPQA, GSM8K, DROP, MGSM, TriviaQA, HumanEval, SimpleQA, HealthBench, PinchBench, XSTest, SWE-bench Verified, SWE-bench Multilingual.

```{toctree}
:maxdepth: 2
:hidden:

get-started/install
get-started/quickstart
tutorials/index
architecture/index
evaluation/benchmarks
evaluation/scoring
deployment/index
api/index
```
