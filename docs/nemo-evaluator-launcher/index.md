# NeMo Evaluator Launcher

NeMo Evaluator Launcher is the user-facing orchestration layer for running AI model evaluations at scale. It provides a unified CLI and programmatic entry points to discover benchmarks, configure runs, submit jobs to different execution backends, monitor progress, and export results.

## How it relates to NeMo Evaluator

- **nemo-evaluator (core engine)**: The evaluation engine that defines the adapter architecture, evaluation workflows, and the ready-to-use evaluation containers). It focuses on correctness, reproducibility, and standardized metrics across benchmarks.
- **nemo-evaluator-launcher (this project)**: An orchestration layer that sits on top of the core engine and containers. It streamlines configuration, execution, and result management across local machines, clusters (e.g., Slurm), CI systems, and hosted backends.

Put simply: the core evaluates, the launcher coordinates.

## Why a separate launcher?

- **Abstraction over containers**: Reuse the open, ready-to-use evaluation containers without hand-crafting commands
- **Scalable execution**: Submit and manage many benchmarks/models across varied backends (local, Slurm, CI, hosted)
- **Configuration management**: [Hydra](https://hydra.cc/docs/intro/)-based configuration with examples and overrides for rapid iteration
- **Reproducibility**: Every run’s fully-resolved config is saved and can be re-run exactly
- **Monitoring and control**: Consistent status, logging, and lifecycle commands regardless of where jobs run
- **Result export**: Support for files, W&B, MLflow, Google Sheets, and more

## Typical workflow

1. Pick execution backend (Local, Slurm, Lepton)
2. Select or copy an example config from the examples directory
3. Point the target to your OpenAI-compatible endpoint (self-hosted or hosted)
4. Launch evaluations via CLI or API
5. Tail logs, check status, and export results

## When to use the launcher

Use the launcher whenever you want:
- A single, consistent command surface for running evaluations anywhere
- To coordinate multiple benchmarks/models concurrently
- Turnkey reproducibility and run bookkeeping 
- Built-in export to popular ML tools (W&B, MLflow, Google Sheets)

For a guided setup, start with the [Quickstart](quickstart.md).

Curious about where and how evaluations run? Explore the available backends in the [Executors Overview](executors/overview.md). Ready to publish results to files, W&B, MLflow, or Sheets? See the [Exporters Overview](exporters/overview.md).


## Disclaimer

This project will download and install additional third-party open source software projects. Review the license terms of these open source projects before use.
