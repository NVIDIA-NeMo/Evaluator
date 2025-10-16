(gs-quickstart-launcher)=
# NeMo Evaluator Launcher

**Best for**: Most users who want a unified CLI experience

The NeMo Evaluator Launcher provides the simplest way to run evaluations with automated container management, built-in orchestration, and comprehensive result export capabilities.

## Prerequisites

- OpenAI-compatible endpoint (hosted or self-deployed) and an API key (if the endpoint is gated)
- Docker installed (for local execution)
- NeMo Evaluator repository cloned (for access to [examples](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples))
- Your Hugging Face token with access to the GPQA-Diamond dataset (click [here](https://huggingface.co/datasets/Idavidrein/gpqa) to request)

## Quick Start

```bash
# 1. Install the launcher
pip install nemo-evaluator-launcher

# Optional: Install with specific exporters
pip install "nemo-evaluator-launcher[mlflow,wandb,gsheets]"  # All exporters
pip install "nemo-evaluator-launcher[mlflow]"                # MLflow only
pip install "nemo-evaluator-launcher[wandb]"                 # W&B only
pip install "nemo-evaluator-launcher[gsheets]"               # Google Sheets only

# 2. List available benchmarks
nemo-evaluator-launcher ls tasks

# 3. Run evaluation against a hosted endpoint

# Prerequisites: Set your API key and HF token
export NGC_API_KEY=nvapi-...
export HF_TOKEN_FOR_GPQA_DIAMOND=hf_...

```

```{literalinclude} ../_snippets/launcher_basic.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

```bash
# 4. Check status
nemo-evaluator-launcher status <invocation_id> --json  # use the ID printed by the run command

# 5. Check the results
cat <job_output_dir>/artifacts/results.yml   # use the output_dir from command above

# 6. Export your results (JSON/CSV)
nemo-evaluator-launcher export <invocation_id> --dest local --format json

# 7. List your recent runs
nemo-evaluator-launcher ls runs --since 2d   # list runs from last 2 days
```

:::{note}
It is possible to use short version of IDs in `status` command, for example `abcd` instead of a full `abcdef0123456` or `ab.0` instead of `abcdef0123456.0`, so long as there are no collisions. This is a syntactic sugar allowing for a slightly easier usage.
:::

## Complete Working Example

Here's a complete example using NVIDIA Build (build.nvidia.com):

```bash
# Prerequisites: Set your API key and HF token
export NGC_API_KEY=nvapi-...
export HF_TOKEN_FOR_GPQA_DIAMOND=hf_...
```

```{literalinclude} ../_snippets/launcher_full_example.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

**What happens:**

- Pulls appropriate evaluation container
- Runs benchmark against your endpoint
- Saves results to specified directory
- Provides monitoring and status updates

## Key Features

### Automated Container Management

- Automatically pulls and manages evaluation containers
- Handles volume mounting for results
- No manual Docker commands required

### Built-in Orchestration

- Job queuing and parallel execution
- Progress monitoring and status tracking

### Result Export

- Export to MLflow, Weights & Biases, or local formats
- Structured result formatting
- Integration with experiment tracking platforms

### Configuration Management

- YAML-based configuration system
- Override parameters via command line
- Template configurations for common scenarios

## Next Steps

- Explore different evaluation types: `nemo-evaluator-launcher ls tasks`
- Try advanced configurations in the `packages/nemo-evaluator-launcher/examples/` directory
- Export results to your preferred tracking platform
- Scale to cluster execution with Slurm or cloud providers

For more advanced control, consider the {ref}`gs-quickstart-core` Python API or {ref}`gs-quickstart-container` approaches.
