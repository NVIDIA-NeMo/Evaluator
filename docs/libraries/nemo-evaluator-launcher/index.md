(lib-launcher)=

# NeMo Evaluator Launcher

NeMo Evaluator Launcher is the user-facing orchestration layer for running AI model evaluations at scale. It provides a unified CLI and programmatic entry points to discover benchmarks, configure runs, submit jobs to different execution backends, monitor progress, and export results.

## Typical workflow

1. Pick execution backend (local, Slurm, CI, hosted)
2. Select or copy an example config from the examples directory
3. Point the target to your OpenAI-compatible endpoint (self-hosted or hosted)
4. Launch evaluations via CLI or API
5. Tail logs, check status, and export results

## When to use the launcher

Use the launcher whenever you want:
- A single, consistent command surface for running evaluations anywhere
- To coordinate multiple benchmarks/models concurrently
- Turnkey reproducibility and run bookkeeping
- Easy integration with exporters and dashboards

For a guided setup, start with the [Quickstart](quickstart.md).

Curious about where and how evaluations run? Explore the available backends in the [Executors Overview](executors/overview.md). Ready to publish results to files, W&B, MLflow, or Sheets? See the [Exporters Overview](exporters/overview.md).
