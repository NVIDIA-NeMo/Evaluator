(gs-quickstart-launcher)=
# NeMo Evaluator Launcher

**Best for**: Most users who want a unified CLI experience

The NeMo Evaluator Launcher provides the simplest way to run evaluations with automated container management, built-in orchestration, and comprehensive result export capabilities.

## Prerequisites

- OpenAI-compatible endpoint (hosted or self-deployed)
- Docker installed (for local execution)

## Quick Start

```bash
# 1. Install the launcher
pip install nemo-evaluator-launcher

# 2. List available benchmarks
nemo-evaluator-launcher ls tasks

# 3. Run evaluation against a hosted endpoint
```

```{literalinclude} ../_snippets/launcher_basic.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```

```bash
# 4. Check status and results
nemo-evaluator-launcher status <invocation_id>
```

## Complete Working Example

Here's a complete example using NVIDIA Build (build.nvidia.com):

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
