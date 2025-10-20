(gs-quickstart-launcher)=
# NeMo Evaluator Launcher

**Best for**: Most users who want a unified CLI experience

The NeMo Evaluator Launcher provides the simplest way to run evaluations with automated container management, built-in orchestration, and comprehensive result export capabilities.

## Prerequisites

- OpenAI-compatible endpoint (hosted or self-deployed) and an API key (if the endpoint is gated), below referred as `NGC_API_KEY` in case one uses models hosted under [NVIDIA's serving platform](https://build.nvidia.com)
- Docker installed (for local execution)
- NeMo Evaluator repository cloned (for access to [examples](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples))
  ```bash
  git clone https://github.com/NVIDIA-NeMo/Evaluator.git
  ```
- Your Hugging Face token with access to the GPQA-Diamond dataset (click [here](https://huggingface.co/datasets/Idavidrein/gpqa) to request), below referred as `HF_TOKEN_FOR_GPQA_DIAMOND`.

## Quick Start

```bash
# 1. Install the launcher
pip install nemo-evaluator-launcher

# Optional: Install with specific exporters
pip install "nemo-evaluator-launcher[all]"      # All exporters
pip install "nemo-evaluator-launcher[mlflow]"   # MLflow only
pip install "nemo-evaluator-launcher[wandb]"    # W&B only
pip install "nemo-evaluator-launcher[gsheets]"  # Google Sheets only

# 2. List available benchmarks
nemo-evaluator-launcher ls tasks

# 3. Run evaluation against a hosted endpoint

# Prerequisites: Set your API key and HF token. Visit https://huggingface.co/datasets/Idavidrein/gpqa
# to get access to the gated GPQA dataset for this task.
export NGC_API_KEY=nvapi-...
export HF_TOKEN_FOR_GPQA_DIAMOND=hf_...
# Move into the cloned directory (see above).
cd Evaluator
```

```{literalinclude} ../_snippets/launcher_basic.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

```bash
# 4. Check status
nemo-evaluator-launcher status <invocation_id> --json  # use the ID printed by the run command

# 5. Find all the recent runs you launched
nemo-evaluator-launcher ls runs --since 2h   # list runs from last 2 hours

```

:::{note}
It is possible to use short version of IDs in `status` command, for example `abcd` instead of a full `abcdef0123456` or `ab.0` instead of `abcdef0123456.0`, so long as there are no collisions. This is a syntactic sugar allowing for a slightly easier usage.
:::

```bash
# 6a. Check the results
cat <job_output_dir>/artifacts/results.yml   # use the output_dir printed by the run command

# 6b. Check the running logs
tail -f <job_output_dir>/*/logs/stdout.log   # use the output_dir printed by the run command

# 7a. Export your results (JSON/CSV)
nemo-evaluator-launcher export <invocation_id> --dest local --format json
# 7b. Or debug them, with lots of useful subcommands inside
nemo-evaluator-launcher debug <invocation_id>   # use the ID printed by the run command

# 8. Kill the running job(s)
nemo-evaluator-launcher kill <invocation_id>  # use the ID printed by the run command
```


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
