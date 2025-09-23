# Configuration

The `nemo-evaluator-launcher` uses [Hydra](https://hydra.cc/docs/intro/) for configuration management, providing a powerful and flexible way to compose configurations from multiple sources and override them from the command line.

## Configuration Structure

The configuration includes these predefined sections:

- **`defaults`** - Specifies default execution and deployment methods
- **`execution`** - Setup for execution configuration ([Execution Overview](execution/index.md) | [Local](execution/local.md), [SLURM](execution/slurm.md), [Lepton](execution/lepton.md))
- **`deployment`** - Setup for deployment configuration ([Deployment Overview](deployment/index.md) | [vLLM](deployment/vllm.md), [SGLang](deployment/sglang.md), [NIM](deployment/nim.md), [None](deployment/none.md))
- **`target`** - API endpoint configuration ([Target Overview](target/index.md))
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

### 1. Defaults

Uses [Hydra's defaults list](https://hydra.cc/docs/advanced/defaults_list/) to compose configuration files. Mix and match execution back ends and deployment methods.

**Available Options:**

- **Execution**: `local`, `slurm`, `lepton`
- **Deployment**: `vllm`, `sglang`, `nim`, `none`

### 2. Execution

Defines how and where to run evaluations. Refer to [Execution Overview](execution/index.md) for details.

- **[Local](execution/local.md)**: Run on your machine with Docker
- **[SLURM](execution/slurm.md)**: Submit jobs to HPC clusters
- **[Lepton](execution/lepton.md)**: Deploy and run on Lepton AI

### 3. Deployment

Defines how to deploy and serve your model. Refer to [Deployment Overview](deployment/index.md) for details.

- **[vLLM](deployment/vllm.md)**: Fast inference and serving for large language models
- **[SGLang](deployment/sglang.md)**: Fast serving framework for large language models and vision-language models
- **[NIM](deployment/nim.md)**: NVIDIA Inference Microservices
- **[None](deployment/none.md)**: Use existing endpoint (no deployment)

### 4. Target

Defines the model endpoint to test. Refer to [Target Overview](target/index.md) for details.

Use this section when you set `deployment: none`. For evaluations that include deployment, the launcher populates this section automatically.

### 5. Evaluation

Defines which benchmarks to run and their configuration. Refer to [Evaluation Overview](evaluation/index.md) for details.

This configuration applies to all executors and you can reuse it across them. It supports both parameter overrides and custom `config` objects for advanced configuration.


## Command-Line Overrides

You can override any configuration value from the command line using [Hydra override syntax](https://hydra.cc/docs/advanced/override_grammar/basic/). This approach lets you test small changes without writing a new configuration file:

```bash
nemo-evaluator-launcher run --config-dir examples --config-name local_llama_3_1_8b_instruct \
  -o execution.output_dir=my_results \
  -o target.api_endpoint.model_id=model/another/one
```

## Configuration Sources

All configuration files are available in the `nemo-evaluator-launcher` repository:

- [Main Configuration Directory](../../../packages/nemo-evaluator-launcher/src/nemo_evaluator_launcher/configs)

## Troubleshooting

### Validate Configuration with a Dry Run

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

### Log Failed Request and Response Pairs

```yaml
evaluation:
  overrides:
    target.api_endpoint.adapter_config.log_failed_requests: true
```

### Log Requests

You can set a limit. By default, the launcher logs all requests when `max_logged_requests` is `null`.

```yaml
evaluation:
  overrides:
    target.api_endpoint.adapter_config.max_logged_requests: null
    target.api_endpoint.adapter_config.use_request_logging: true
```


### Log Responses

You can set a limit. By default, the launcher logs all responses when `max_logged_responses` is `null`.

```yaml
evaluation:
  overrides:
    target.api_endpoint.adapter_config.max_logged_responses: null
    target.api_endpoint.adapter_config.use_response_logging: true
```


### Test with Limited Samples

Start with a small sample size to verify that your configuration works:

```yaml
evaluation:
  overrides:
    config.params.limit_samples: 10  # Test with only 10 samples
```

### Check Endpoint Availability

Verify your endpoint is working before running full evaluations:

- **Test endpoint compatibility**: Refer to [Testing Endpoint OAI Compatibility](../tutorials/deployments/testing-endpoint-oai-compatibility.md)
- **Check API keys**: Ensure environment variables are set correctly

### Common Issues

- **Configuration errors**: Use `--dry-run` to verify the configuration before execution
- **Endpoint timeouts**: Increase `request_timeout` in your configuration
- **Memory issues**: Reduce `parallelism` or `limit_samples` for testing
- **API key problems**: Verify environment variable names match your configuration
- **Hydra override syntax**: Refer to [Hydra override grammar](https://hydra.cc/docs/advanced/override_grammar/basic/) for correct syntax

## Reference

- **Debug Logging**: [Logging Configuration](../../nemo-evaluator/reference/logging.md) - Detailed logging setup and troubleshooting
- **Parameter Overrides**: [nemo-evaluator CLI Reference](../../nemo-evaluator/reference/cli.md#parameter-overrides) - Complete guide to available parameters and override syntax
