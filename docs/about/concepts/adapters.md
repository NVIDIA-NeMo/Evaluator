(adapters-concepts)=
# Adapters

Adapters in NeMo Evaluator provide sophisticated request and response processing through a configurable interceptor pipeline. They enable advanced evaluation capabilities like caching, logging, reasoning extraction, and custom prompt injection.

## Architecture Overview

The adapter system transforms simple API calls into sophisticated evaluation workflows:

1. **Request Processing**: Interceptors modify outgoing requests (system prompts, parameters)
2. **API Communication**: Endpoint interceptor handles HTTP communication with retries
3. **Response Processing**: Interceptors extract reasoning, log data, and cache results
4. **Result Integration**: Processed responses integrate with evaluation frameworks

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
- **Progress Tracking**: Monitor evaluation progress
- **Endpoint**: Handle HTTP communication with the model API

## Integration

The adapter system integrates seamlessly with:
- **Evaluation Frameworks**: Works with any OpenAI-compatible API
- **NeMo Evaluator Core**: Direct integration via `AdapterConfig`
- **NeMo Evaluator Launcher**: YAML configuration support

## Usage Example

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    api_url="http://localhost:8080/v1/completions/",
    use_reasoning=True,
    use_caching=True,
    use_request_logging=True,
    custom_system_prompt="You are a helpful assistant."
)
```

For detailed usage and configuration examples, see [Adapters & Interceptors](adapters-interceptors.md).


