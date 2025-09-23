# Execution Configuration

Execution configurations define how and where to run evaluation tasks.

## Executors

- **[Local](local.md)**: Development, testing, and evaluation of already deployed endpoints (Docker-based)
- **[SLURM](slurm.md)**: HPC clusters and large-scale evaluations (can also deploy models)
- **[Lepton](lepton.md)**: Cloud deployments and parallel evaluations (can also deploy models)

## Quick Reference

```yaml
execution:
  type: local  # or slurm, lepton
  output_dir: /path/to/results  # Common: where to save results (mounted as /results in evaluation container, must be creatable on the given executor)
  auto_export:                   # Common: automatic result export
    destinations: ["wandb", "mlflow", "gsheets", "s3"]
    configs:
      wandb: {}
      mlflow: {}
      gsheets: {}
      s3: {}
  # Executor-specific parameters (see individual executor docs):
```

## Auto Export

At the executor level, define result export methods. The `auto_export` field enables automatic export of results to the following platforms:

- **Weights & Biases (W&B)**: Experiment tracking
- **MLflow**: Experiment management
- **Google Sheets**: Spreadsheet integration
- **Amazon S3**: Object storage

For detailed configuration options, refer to the {doc}`Exporters Overview <../../exporters/overview>`.

For detailed executor information, refer to the [Executors Overview](../../executors/overview.md).
