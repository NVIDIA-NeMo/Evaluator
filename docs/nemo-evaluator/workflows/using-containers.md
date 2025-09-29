(using-containers)=
# Container Workflows

This document explains how NeMo Evaluator workflows use evaluation containers, from selection and execution to integration with the CLI launcher and pipeline system.

## Overview

Evaluation containers are the execution environments that run benchmarks and evaluations in NeMo Evaluator. They provide consistent, reproducible environments for running different types of AI model evaluations, ensuring consistent results across different runs and systems.

## Available Container Types

For a comprehensive list of all available evaluation factory containers with detailed descriptions, specifications, and usage examples, refer to the [Container Reference](nemo-evaluator/reference/containers.md).

## Evaluation Execution Workflow

### Command Structure

The system uses the `eval-factory` command with extensive configuration options:

```bash
docker run --rm -it nvcr.io/nvidia/eval-factory/simple-evals:25.08.1 bash

export HF_TOKEN=hf_xxx # Supply HF TOKEN
export MY_API_KEY=nvapi-xxx # API_KEY for build.nvidia.com access

eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir `mktemp -d` \
    --overrides 'config.params.limit_samples=3'
```

The example above evaluates the `meta/llama-3.1-8b-instruct` model on the `mmlu_pro` benchmark from the `simple-evals` framework. The `limit_samples` parameter restricts evaluation to three samples instead of the entire dataset.

**Quick Overview:**
NeMo Evaluator Launcher provides specialized containers for different evaluation domains including language models, code generation, vision-language models, agent AI, retrieval systems, and safety evaluation. Each container targets specific use cases and comes with pre-configured evaluation harnesses.

## Adapter-Based Execution

## Adapter System Architecture

The system uses an interceptor-based architecture that processes requests and
responses through a chain of adapters. The system includes three primary types of interceptors:

- **RequestInterceptor**: Captures and modifies requests sent to the endpoint, allowing for customization of system messages, request parameters, and more.
- **ResponseInterceptor**: Captures and processes responses received from the endpoint, enabling functionalities such as token usage tracking, removal of reasoning tokens, and more.
- **Post-eval hook**: Runs after the evaluation completes, allowing for cleanup, report generation, or other post-processing tasks.

**Example**: To benchmark the `nvidia/llama-3.3-nemotron-super-49b-v1.5` model, use the `system_message` interceptor to include the `/think` system message and activate reasoning mode. Then, apply the `reasoning` interceptor to remove reasoning tokens before evaluating the model's responses.

Interceptors enable straightforward evaluation of endpoints that:

- require extra parameters
- do not support certain parameters
- need custom system messages
- require reasoning handling
and more, making the workflow highly adaptable

**Configuration Methods**  
You have two options for specifying adapters:

- **CLI Overrides**: Use `--overrides` parameter for runtime configuration ([learn more](../reference/api.md#interceptor-system))
- **YAML Configuration**: Define interceptor chains in configuration files ([learn more](../reference/api.md#interceptor-system))

## Configuration and Overrides

## CLI Parameter Overrides

Each benchmark supported by NeMo Evaluator comes with its own set of predefined, default parameters. The system supports extensive configuration through the `--overrides` parameter:

```bash
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir `mktemp -d` \
    --overrides 'config.params.limit_samples=3,config.params.temperature=0.5,target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_caching=True'
```

In this example, several parameter overrides customize the evaluation:

- The `temperature` parameter overrides the default value with 0.5
- The `limit_samples` parameter restricts the evaluation to just 3 samples rather than the full dataset
- Two adapters activate:
  - `use_request_logging` to capture and log the requests sent to the model (or to the next adapter)
  - `use_caching` to cache model's responses

## Workflow Examples

### Basic Evaluation Workflow

```bash
# Run evaluation with eval-factory
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results
```

### Advanced Configuration Workflow

```bash
# 1. Enable comprehensive logging and caching
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_response_logging=True,target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache'

# 2. The system provides enhanced monitoring and performance
# → All requests and responses are logged
# → Caching improves performance for repeated evaluations
# → Detailed logs are available for analysis
```

For more details, refer to the [CLI](../reference/cli.md).

## Performance and Monitoring

### Caching and Performance

The system provides built-in performance optimization through caching:

- **Response Caching**: Caches model responses for reuse
- **Disk Storage**: Persistent caching with configurable directories
- **Performance Monitoring**: Track request/response patterns

### Logging and Debugging

The system also provides comprehensive logging capabilities for monitoring and troubleshooting:

- **Request Logging**: Log all requests sent to models
- **Response Logging**: Log all model responses
- **Failed Request Logging**: Track and debug failed requests
- **HTML Report Generation**: Generate detailed reports of cached data

## Troubleshooting

### Port Conflicts

If you encounter port conflicts, you can change the adapter server port:

```bash
export ADAPTER_PORT=3828
```

### Configuration Issues

Enable verbose logging to debug configuration issues:

```bash
--overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.log_failed_requests=True'
```

### API Key Issues

Verify that you set your API key in the environment:

```bash
export MY_API_KEY=your_api_key_here
```

## Best Practices

### 1. Use Environment Variables

- Store sensitive information like API keys in environment variables
- Use consistent naming conventions for environment variables
- Document required environment variables for your team

### 2. Test Configurations

- Start with small sample sizes for testing (for example, `config.params.limit_samples=10`)
- Verify configurations work before running large evaluations
- Use the `--overrides` parameter to test different settings

### 3. Enable Logging and Caching

- Use request and response logging for debugging
- Enable caching to resume failed evaluations
- Generate HTML reports for detailed analysis

### 4. Track Progress

- Check logs frequently for any issues
- Track cache usage and performance

### 5. Configuration Management

- Use consistent configuration patterns across evaluations
- Document your configuration overrides
- Version control your configuration files

## Environment Variables

### Adapter Server Configuration

You can configure the adapter server using environment variables:

```bash
export ADAPTER_PORT=3828
export ADAPTER_HOST=localhost
```

### API Key Management

Store your API keys securely in environment variables:

```bash
export MY_API_KEY=your_api_key_here
export HF_TOKEN=your_hf_token_here
```
