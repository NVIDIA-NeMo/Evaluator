(telemetry)=

# Telemetry

NeMo Evaluator collects telemetry to help improve the project.

**All telemetry events are collected anonymously.**

## Telemetry Levels

NeMo Evaluator supports three telemetry levels:

```{list-table}
:header-rows: 1
:widths: 15 25 60

* - Level
  - Name
  - Description
* - `0`
  - OFF
  - No telemetry reporting at all.
* - `1`
  - MINIMAL
  - Usage metrics are reported but `model_id` is **redacted**.
* - `2` (default)
  - DEFAULT
  - Full reporting including `model_id`.
```

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
  - Model name used for evaluation (redacted at level 1).
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
  - Model name for the launched run (redacted at level 1).
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
* - `NEMO_EVALUATOR_TELEMETRY_LEVEL=0`
  - Disables telemetry globally.
* - `NEMO_EVALUATOR_TELEMETRY_LEVEL=1`
  - Enables telemetry without sending `model_id`.
* - `NEMO_EVALUATOR_TELEMETRY_LEVEL=2`
  - Enables full telemetry (default).
* - `nemo-evaluator-launcher config set telemetry.level <0|1|2>`
  - Persists telemetry level to the config file. See {ref}`configuration`.
```

**Priority:** environment variable > config file > default (2).

## Aggregate Reporting

We may share aggregated telemetry trends with the community (for example, popularity of models, tasks, and execution backends). Aggregates are anonymous and are not used to track individual users.
