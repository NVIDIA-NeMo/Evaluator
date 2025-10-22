# Execution Configuration

Execution configurations define how and where to run evaluation tasks.

## Executors

- **[Local](local.md)**: Development, testing, and evaluation of already deployed endpoints (Docker-based)
- **[Slurm](slurm.md)**: HPC clusters and large-scale evaluations (can also deploy models)
- **[Lepton](lepton.md)**: Cloud deployments and parallel evaluations (can also deploy models)

## Quick Reference

```yaml
execution:
  type: local  # or slurm, lepton
  output_dir: /path/to/results  # Common: where to save results (mounted as /results in evaluation container, must be creatable on the given executor)
  auto_export:                   # Common: automatic result export
    destinations: ["wandb", "mlflow", "gsheets"]
    configs:
      wandb: {}
      mlflow: {}
      gsheets: {}
  # Executor-specific parameters (see individual executor docs):
```

## Auto Export

On the executor level, we can define results export methods. The `auto_export` field enables automatic result export to various platforms:

- **W&B**: Weights & Biases experiment tracking
- **MLFlow**: ML experiment management
- **GSheets**: Google Sheets integration

For detailed configuration options, see the {doc}`Exporters Documentation <../../exporters/overview>`.

For detailed executor information, see the [Executors Overview](../../executors/overview.md).

