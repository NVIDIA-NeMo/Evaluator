# Local Executor

The Local executor runs evaluations on your machine using Docker. It provides a fast way to iterate when you have Docker installed and want fully reproducible runs.

See common concepts and commands in the [executors overview](overview.md).

## Prerequisites
- Docker installed and running
- Python environment with the Nemo Evaluator Launcher CLI available (install the launcher by following the [Quickstart](../quickstart.md))

## Quick Start (Local)

Run a local evaluation using an example configuration:

```bash
# Using short alias (recommended)
nv-eval run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>

# Or using full command name
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  --override execution.output_dir=<YOUR_OUTPUT_LOCAL_DIR>
```

The CLI prints an invocation ID and useful log commands.

### Live Log Monitoring
Use `tail -f` to follow logs:
```bash
tail -f /path/to/output/<INVOCATION_ID>/*/logs/stdout.log
```

Create a custom configuration by copying an example and editing values:
```bash
mkdir -p my_configs
cp examples/local_llama_3_1_8b_instruct.yaml my_configs/my_evaluation.yaml
# edit my_configs/my_evaluation.yaml
```

Run your custom configuration:
```bash
nv-eval run --config-dir my_configs --config-name my_evaluation
```
