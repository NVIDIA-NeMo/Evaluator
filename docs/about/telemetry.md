(telemetry)=

# Telemetry

NeMo Evaluator collects telemetry to help improve the project.

**All telemetry events are collected anonymously.**

## Event: `EvaluationTaskEvent` (from `nemo-evaluator`)

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Description
* - `task`
  - Evaluated task/benchmark name.
* - `frameworkName`
  - Evaluation framework name (for example `lm-eval`, `helm`).
* - `model`
  - Model name used for evaluation.
* - `executionDurationSeconds`
  - Evaluation duration in seconds.
* - `status`
  - Task status: `started`, `success`, or `failure`.
```

## Event: `LauncherJobEvent` (from `nemo-evaluator-launcher`)

```{list-table}
:header-rows: 1
:widths: 30 70

* - Field
  - Description
* - `executorType`
  - Launcher executor backend (`local`, `slurm`, `lepton`, etc.).
* - `deploymentType`
  - Deployment type (`none`, `vllm`, `sglang`, `nim`, etc.).
* - `model`
  - Model name for the launched run.
* - `tasks`
  - List of requested evaluation tasks.
* - `exporters`
  - List of configured exporters.
* - `status`
  - Job status: `started`, `success`, or `failure`.
```

## Telemetry Controls

```{list-table}
:header-rows: 1
:widths: 40 60

* - Control
  - Effect
* - `NEMO_EVALUATOR_TELEMETRY_ENABLED=false`
  - Disables telemetry globally.
* - `nemo-evaluator-launcher --no-telemetry` (or `-T`)
  - Disables telemetry for that launcher invocation.
```

## Aggregate Reporting

We may share aggregated telemetry trends with the community (for example, popularity of models, tasks, and execution backends). Aggregates are anonymous and are not used to track individual users.
