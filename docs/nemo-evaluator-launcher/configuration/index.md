# Configuration

The nemo-evaluator-launcher uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, providing a powerful and flexible way to compose configurations from multiple sources and override them via command line.

## Configuration Structure

The configuration is organized into these predefined sections:

- **`defaults`** - Specifies default execution and deployment methods
- **`execution`** - Additional setup for execution configuration ([Execution Overview](execution/index.md) | [Local](execution/local.md), [Slurm](execution/slurm.md), [Lepton](execution/lepton.md))
- **`deployment`** - Additional setup for deployment configuration ([Deployment Overview](deployment/index.md) | [vLLM](deployment/vllm.md), [SGLang](deployment/sglang.md), [NIM](deployment/nim.md), [None](deployment/none.md))
- **`target`** - API endpoint configuration ([Target Overview](nemo-evaluator-launcher/configuration/target/index.md))
- **`evaluation`** - Evaluation tasks from nemo-evaluator-launcher with additional setup like overriding default values ([Evaluation Overview](evaluation/index.md))

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

# 1. Defaults
Uses [Hydra's defaults list](https://hydra.cc/docs/advanced/defaults_list/) to compose configuration files. Mix and match execution backends and deployment methods.

**Available Options:**
- **Execution**: `local`, `slurm`, `lepton`
- **Deployment**: `vllm`, `sglang`, `nim`, `none`

# 2. Execution
Defines how and where to run evaluations. See [Execution Overview](execution/index.md) for details.

- **[Local](execution/local.md)**: Run on your machine with Docker
- **[Slurm](execution/slurm.md)**: Submit jobs to HPC clusters
- **[Lepton](execution/lepton.md)**: Deploy and run on Lepton AI

# 3. Deployment
Defines how to deploy and serve your model. See [Deployment Overview](deployment/index.md) for details.

- **[vLLM](deployment/vllm.md)**: Fast LLM inference and serving
- **[SGLang](deployment/sglang.md)**: Fast serving framework for LLMs and VLMs
- **[NIM](deployment/nim.md)**: NVIDIA Inference Microservices
- **[None](deployment/none.md)**: Use existing endpoint (no deployment)

# 4. Target
Defines the model endpoint to evaluate. See [Target Overview](nemo-evaluator-launcher/configuration/target/index.md) for details.

Used when `deployment: none` is specified. For evaluations with deployment, this is filled automatically.

# 5. Evaluation
Defines which benchmarks to run and their configuration. See [Evaluation Overview](evaluation/index.md) for details.

Common for all executors and can be reused between them.


## Command Line Overrides 

You can override any configuration value from the command line using [Hydra's override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). This allows you to test small changes without the necessity of writing a new config file:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

## Configuration Sources

All configuration files are available in the nemo-evaluator-launcher repository:
- [Main Config Directory](../../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs)

## Troubleshooting

# Validate Configuration with Dry Run
Always test your configuration before running evaluations:

```bash
# Preview your job configuration without executing
nemo-evaluator-launcher run --config-dir configs --config-name your_config --dry-run

# This shows the complete resolved configuration and scripts that would be generated
```

# Debug Mode
Enable debug logging for detailed error information and troubleshooting:

```bash
# Set environment variable (recommended)
export NEMO_EVALUATOR_LOG_LEVEL=DEBUG

# Run your evaluation
nemo-evaluator-launcher run --config-name your_config
```

# Test with Limited Samples
Start with a small sample size to verify your configuration works:

```yaml
evaluation:
  overrides:
    config.params.limit_samples: 10  # Test with only 10 samples
```

# Check Endpoint Availability
Verify your endpoint is working before running full evaluations:

- **Test endpoint compatibility**: See [Testing Endpoint OAI Compatibility](../tutorials/deployments/testing-endpoint-oai-compatibility.md)
- **Check API keys**: Ensure environment variables are set correctly

# Common Issues
- **Configuration errors**: Use `--dry-run` to validate before execution
- **Endpoint timeouts**: Increase `request_timeout` in your configuration
- **Memory issues**: Reduce `parallelism` or `limit_samples` for testing
- **API key problems**: Verify environment variable names match your configuration
- **Hydra override syntax**: Check [Hydra's override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for correct syntax

# Reference
- **Debug Logging**: [Logging Configuration](../../nemo-evaluator/reference/logging.md) - Detailed logging setup and troubleshooting
- **Parameter Overrides**: [nemo-evaluator CLI Reference](../../nemo-evaluator/reference/cli.md#parameter-overrides) - Complete guide to available parameters and override syntax
