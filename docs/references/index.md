(references-overview)=

# References

Comprehensive reference documentation for NeMo Evaluator APIs, functions, and configuration options.


## Prerequisites

- **Container way**: Use evaluation containers mentioned in {ref}`nemo-evaluator-containers`
- **Package way**:

  ```bash
  pip install nemo-evaluator
  ```

  To run evaluations, you also need to install an evaluation framework package (for example, `nvidia-simple-evals`):
  ```bash
  pip install nvidia-simple-evals
  ```

## CLI vs. Programmatic Usage

The NeMo Evaluator API supports two usage patterns:

1. **CLI Usage** (Recommended): Use `nemo-evaluator run_eval` function which parses command line arguments
2. **Programmatic Usage**: Use `evaluate()` function with configuration objects

**When to Use Which:**

- **CLI**: For command-line tools, scripts, and simple automation
- **Programmatic**: For building custom applications, workflows, and integration with other systems

## API References

::::{grid} 1 2 2 2
:gutter: 1 1 1 2


:::{grid-item-card} {octicon}`command-palette;1.5em;sd-mr-1` NeMo Evaluator Launcher CLI
:link: ../libraries/nemo-evaluator-launcher/cli
:link-type: doc
Comprehensive command-line interface reference with all commands, options, and examples.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` NeMo Evaluator Launcher API
:link: ../libraries/nemo-evaluator-launcher/api
:link-type: doc
Complete Python API reference for programmatic evaluation workflows and job management.
:::

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Configuration Schema
:link: ../libraries/nemo-evaluator-launcher/configuration/index
:link-type: doc
Configuration reference for NeMo Evaluator Launcher with examples for all executors and deployment types.
:::


:::{grid-item-card} {octicon}`command-palette;1.5em;sd-mr-1` NeMo Evaluator CLI
:link: api/nemo-evaluator/cli
:link-type: doc
Comprehensive command-line interface reference with all commands, options, and examples.
:::

:::{grid-item-card} {octicon}`terminal;1.5em;sd-mr-1` NeMo Evaluator Python API
:link: api/nemo-evaluator/api/index
:link-type: doc
Complete Python API reference for programmatic evaluation workflows and job management.
:::

::::
