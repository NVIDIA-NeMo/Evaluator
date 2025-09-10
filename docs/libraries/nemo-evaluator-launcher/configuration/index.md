(configuration-overview)=

# Configuration

The nemo-evaluator-launcher uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, providing a powerful and flexible way to compose configurations from multiple sources and override them via command line.

## Configuration Structure

The configuration is organized into these predefined sections:

- **`defaults`** - Specifies default execution and deployment methods
- **`execution`** - Execution backend configuration (see [Executors Overview](../executors/overview.md))
- **`deployment`** - Model deployment configuration ([Deployment Overview](deployment/index.md))
- **`target`** - API endpoint configuration ([Target Overview](target/index.md))
- **`evaluation`** - Evaluation tasks configuration ([Evaluation Overview](evaluation/index.md))

```yaml
defaults:
  - execution: local  # or slurm, lepton
  - deployment: none  # or vllm, sglang, nim
  - _self_

execution:
  # executor-specific settings...

deployment:
  # deployment-specific settings...

target:
  api_endpoint:
    model_id: your-model-name  # example: meta/llama-3.1-8b-instruct
    url: https://your-endpoint.com/v1/chat/completions  # example: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY  # example: API_KEY

# specify the benchmarks to evaluate
evaluation:
  overrides:
    # global overrides for all tasks
  tasks:
    - name: task_name
    - name: another_task
      overrides:
        # task-specific overrides
      env_vars:
        # task-specific environment variables
```

## Configuration Sections

Learn how to configure and customize NeMo Evaluator Launcher for your specific needs:

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`gear;1.5em;sd-mr-1` Execution
:link: ../executors/overview
:link-type: doc

Defines how and where to run evaluations: Local (Docker), Slurm (HPC clusters), or Lepton (cloud).
:::

:::{grid-item-card} {octicon}`rocket;1.5em;sd-mr-1` Deployment
:link: deployment/index
:link-type: doc

Model deployment configuration: vLLM, SGLang, NIM, or None (existing endpoints).
:::

:::{grid-item-card} {octicon}`location;1.5em;sd-mr-1` Target
:link: target/index
:link-type: doc

API endpoint configuration when using existing model endpoints (deployment: none).
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Evaluation
:link: evaluation/index
:link-type: doc

Benchmark selection and task configuration with parameter overrides and environment variables.
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


## Command Line Overrides 

You can override any configuration value from the command line using [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). This allows you to test small changes without the necessity of writing a new config file:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

## Configuration Sources

All configuration files are available in the nemo-evaluator-launcher repository:
- [Main Config Directory](https://gitlab-master.nvidia.com/dl/JoC/competitive_evaluation/nv-eval-platform/-/tree/main/nemo_evaluator_launcher/src/nemo_evaluator_launcher/configs?ref_type=heads)

## Troubleshooting

### Validate Configuration with Dry Run
Always test your configuration before running evaluations:

```bash
# Preview your job configuration without executing
nemo-evaluator-launcher run --config-dir configs --config-name your_config --dry-run

# This shows the complete resolved configuration and scripts that would be generated
```

### Debug Mode
Enable debug logging for detailed error information and troubleshooting:

```bash
# Set environment variable (recommended)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Run your evaluation
nemo-evaluator-launcher run --config-name your_config
```

### Test with Limited Samples
Start with a small sample size to verify your configuration works:

```yaml
evaluation:
  overrides:
    config.params.limit_samples: 10  # Test with only 10 samples
```

### Check Endpoint Availability
Verify your endpoint is working before running full evaluations:

- **Test endpoint compatibility**: See [Testing Endpoint OAI Compatibility](../tutorials/deployments/testing_endpoint_oai_compatibility.md)
- **Check API keys**: Ensure environment variables are set correctly

### Common Issues
- **Configuration errors**: Use `--dry-run` to validate before execution
- **Endpoint timeouts**: Increase `request_timeout` in your configuration
- **Memory issues**: Reduce `parallelism` or `limit_samples` for testing
- **API key problems**: Verify environment variable names match your configuration
- **Hydra override syntax**: Check [Hydra's override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for correct syntax

### Reference
- **Debug Logging**: [Logging Configuration](../../nemo-evaluator/reference/logging.md) - Detailed logging setup and troubleshooting
- **Parameter Overrides**: [nemo-evaluator CLI Reference](../../nemo-evaluator/reference/cli.md#parameter-overrides) - Complete guide to available parameters and override syntax

```{toctree}
:caption: Configuration Reference
:hidden:

Deployment <deployment/index>
Structure & Sections <structure>
Examples <examples>
Overrides & Composition <overrides>
Parameter Reference <parameters>
Advanced Patterns <advanced>
Validation & Debugging <validation>
```
