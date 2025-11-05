---
orphan: true
---

(adapters-concepts)=
# Adapters

Adapters in NeMo Evaluator provide sophisticated request and response processing through a configurable interceptor pipeline. They enable advanced evaluation capabilities like caching, logging, reasoning extraction, and custom prompt injection.

## Architecture Overview

The adapter system transforms simple API calls into sophisticated evaluation workflows through a two-phase pipeline:

1. **Request Processing**: Interceptors modify outgoing requests (system prompts, parameters) before they reach the endpoint
2. **Response Processing**: Interceptors extract reasoning, log data, cache results, and track statistics after receiving responses

The endpoint interceptor bridges these phases by handling HTTP communication with the model API.

## Core Components

- **AdapterConfig**: Configuration class for all interceptor settings
- **Interceptor Pipeline**: Modular components for request/response processing
- **Endpoint Management**: HTTP communication with error handling and retries
- **Resource Management**: Caching, logging, and progress tracking

## Available Interceptors

The adapter system includes several built-in interceptors:

- **System Message**: Inject custom system prompts
- **Payload Modifier**: Transform request parameters
- **Request/Response Logging**: Capture detailed interaction data
- **Caching**: Store and retrieve responses for efficiency
- **Reasoning**: Extract chain-of-thought reasoning
- **Response Stats**: Collect aggregated statistics from API responses
- **Progress Tracking**: Monitor evaluation progress
- **Endpoint**: Handle HTTP communication with the model API
- **Raise Client Errors**: Handle and raise exceptions for client errors

## Integration

The adapter system integrates seamlessly with:

- **Evaluation Frameworks**: Works with any OpenAI-compatible API
- **NeMo Evaluator Core**: Direct integration via `AdapterConfig`
- **NeMo Evaluator Launcher**: YAML configuration support

## Configuration

### Modern Interceptor-Based Configuration

The recommended approach uses the interceptor-based API:

:::{code-block} python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            enabled=True,
            config={"system_message": "You are a helpful assistant."}
        ),
        InterceptorConfig(name="request_logging", enabled=True),
        InterceptorConfig(
            name="caching",
            enabled=True,
            config={"cache_dir": "./cache", "reuse_cached_responses": True}
        ),
        InterceptorConfig(name="reasoning", enabled=True),
        InterceptorConfig(name="endpoint")
    ]
)
:::

For detailed usage and configuration examples, see {ref}`interceptors-concepts`.
