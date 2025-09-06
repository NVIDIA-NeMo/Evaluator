(configuration-overview)=

# Configuration Reference

NeMo Evaluator Launcher uses [Hydra](https://hydra.cc/) for configuration management, allowing for flexible, composable, and overridable configurations.

## Configuration Structure

All configurations follow this general structure:

```yaml
# Specify default sub-configurations
defaults:
  - execution: local          # Execution backend (local, slurm, lepton)
  - deployment: none          # Model deployment (none, vllm, nim, sglang)
  - _self_                    # Include this config's values

# Execution settings
execution:
  output_dir: results         # Where to store results

# Model endpoint configuration  
target:
  api_endpoint:
    url: http://localhost:8000/v1/chat/completions
    model_id: my-model
    api_key_name: API_KEY     # Environment variable name

# Evaluation configuration
evaluation:
  overrides:                  # Global overrides for all tasks
    config.params.temperature: 0.7
  tasks:                      # List of evaluation tasks
    - name: hellaswag
    - name: arc_challenge
```

## Core Configuration Topics

Learn how to configure and customize NeMo Evaluator Launcher for your specific needs:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Structure & Sections
:link: structure
:link-type: doc

Core configuration sections including defaults, execution, target, evaluation, and deployment settings.
:::

:::{grid-item-card} {octicon}`code-square;1.5em;sd-mr-1` Examples
:link: examples
:link-type: doc

Complete configuration examples for local development, production clusters, and cloud deployments.
:::

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Overrides & Composition
:link: overrides
:link-type: doc

Command-line overrides, environment variables, and configuration composition patterns.
:::

:::{grid-item-card} {octicon}`book;1.5em;sd-mr-1` Parameter Reference
:link: parameters
:link-type: doc

Detailed reference for all task parameters, adapter configuration, and request settings.
:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Advanced Patterns
:link: advanced
:link-type: doc

Real-world configuration patterns including auto-export, metadata injection, and testing workflows.
:::

:::{grid-item-card} {octicon}`bug;1.5em;sd-mr-1` Validation & Debugging
:link: validation
:link-type: doc

Configuration validation, debugging commands, and common error resolution.
:::

::::

## Quick Reference

**Available Execution Backends:**
- `local` - Docker-based evaluation on your workstation
- `slurm` - HPC cluster execution with resource management
- `lepton` - Cloud execution with on-demand GPU provisioning

**Available Deployment Types:**
- `none` - Use external API endpoint
- `vllm` - Deploy models with vLLM serving
- `nim` - Deploy NVIDIA Inference Microservices
- `sglang` - Deploy with SGLang serving framework

## See Also

- [Quickstart](../quickstart.md) - Getting started guide
- [CLI Reference](../cli.md) - Command-line interface
- [Executors](../executors/index.md) - Execution backend options
- [Python API](../api.md) - Programmatic configuration management

```{toctree}
:caption: Configuration Reference
:hidden:

Structure & Sections <structure>
Examples <examples>
Overrides & Composition <overrides>
Parameter Reference <parameters>
Advanced Patterns <advanced>
Validation & Debugging <validation>
```
