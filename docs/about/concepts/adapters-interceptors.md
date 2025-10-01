(adapters-interceptors-concepts)=

# Adapters and Interceptors

The **adapter and interceptor system** is a core architectural concept in NeMo Evaluator that enables sophisticated request and response processing during model evaluation.

## Conceptual Overview

The adapter system transforms simple model API calls into sophisticated evaluation workflows through a configurable pipeline of **interceptors**. This design provides:

- **Modularity**: Each interceptor handles a specific concern (logging, caching, reasoning)
- **Composability**: Multiple interceptors can be chained together
- **Configurability**: Interceptors can be enabled/disabled and configured independently
- **Extensibility**: Custom interceptors can be added for specialized processing

The following diagram shows a typical interceptor pipeline configuration. Note that interceptors must follow the order: Request → RequestToResponse → Response, but the specific interceptors and their configuration are flexible:

```{mermaid}
graph LR
    A[Evaluation Request] --> B[Adapter System]
    B --> C[Interceptor Pipeline]
    C --> D[Model API]
    D --> E[Response Pipeline]
    E --> F[Processed Response]
    
    subgraph "Request Processing"
        C --> G[System Message]
        G --> H[Payload Modifier]
        H --> I[Request Logging]
        I --> J[Caching Check]
        J --> K[Endpoint Call]
    end
    
    subgraph "Response Processing"
        E --> L[Response Logging]
        L --> M[Reasoning Extraction]
        M --> N[Progress Tracking]
        N --> O[Cache Storage]
    end
    
    style B fill:#f3e5f5
    style C fill:#e1f5fe
    style E fill:#e8f5e8
```

## Core Concepts

### Adapters

**Adapters** are the orchestration layer that manages the interceptor pipeline. They provide:

- **Configuration Management**: Unified interface for interceptor settings
- **Pipeline Coordination**: Manages the flow of requests through interceptors
- **Resource Management**: Handles shared resources like caches and logs
- **Error Handling**: Provides consistent error handling across interceptors

### Interceptors

**Interceptors** are modular components that process requests and responses. Key characteristics:

- **Dual Interface**: Each interceptor can process both requests and responses
- **Context Awareness**: Access to evaluation context (benchmark type, model info)
- **Stateful Processing**: Can maintain state across requests
- **Chainable**: Multiple interceptors work together in sequence

## Interceptor Categories

### Processing Interceptors
Transform or augment requests and responses:
- **System Message**: Inject custom prompts
- **Payload Modifier**: Modify request parameters
- **Reasoning**: Extract chain-of-thought reasoning

### Infrastructure Interceptors
Provide supporting capabilities:
- **Caching**: Store and retrieve responses
- **Logging**: Capture request/response data
- **Progress Tracking**: Monitor evaluation progress
- **Response Stats**: Track request statistics and metrics
- **Raise Client Error**: Raise exceptions for client errors (4xx status codes, excluding 408 and 429)

### Integration Interceptors
Handle external system integration:
- **Endpoint**: Route requests to model APIs

## Configuration Philosophy

The adapter system follows a **configuration-over-code** philosophy:

### Simple Configuration
Enable basic features with minimal configuration:
:::{code-block} python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(name="caching", enabled=True),
        InterceptorConfig(name="request_logging", enabled=True),
        InterceptorConfig(name="endpoint")
    ]
)
:::

### Advanced Configuration
Full control over interceptor behavior:
:::{code-block} python
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            enabled=True,
            config={"system_message": "You are an expert."}
        ),
        InterceptorConfig(
            name="caching",
            enabled=True,
            config={"cache_dir": "./cache"}
        ),
        InterceptorConfig(
            name="request_logging",
            enabled=True,
            config={"max_requests": 1000}
        ),
        InterceptorConfig(
            name="reasoning",
            enabled=True,
            config={
                "start_reasoning_token": "<think>",
                "end_reasoning_token": "</think>"
            }
        ),
        InterceptorConfig(name="endpoint")
    ]
)
:::

### YAML Configuration
Declarative configuration for reproducibility:
```yaml
adapter_config:
  interceptors:
    - name: system_message
      enabled: true
      config:
        system_message: "Think step by step."
    - name: caching
      enabled: true
    - name: reasoning
      enabled: true
    - name: endpoint
```

## Design Benefits

### 1. **Separation of Concerns**
Each interceptor handles a single responsibility, making the system easier to understand and maintain.

### 2. **Reusability**
Interceptors can be reused across different evaluation scenarios and benchmarks.

### 3. **Testability**
Individual interceptors can be tested in isolation, improving reliability.

### 4. **Performance**
Interceptors can be optimized independently and disabled when not needed.

### 5. **Extensibility**
New interceptors can be added without modifying existing code.

## Common Use Cases

### Research Workflows
- **Reasoning Analysis**: Extract and analyze model reasoning patterns
- **Prompt Engineering**: Test different system prompts systematically
- **Behavior Studies**: Log detailed interactions for analysis

### Production Evaluations
- **Performance Optimization**: Cache responses to reduce API costs
- **Monitoring**: Track evaluation progress and performance metrics
- **Compliance**: Maintain audit trails of all interactions

### Development and Debugging
- **Request Inspection**: Log requests to debug evaluation issues
- **Response Analysis**: Capture detailed response data
- **Error Tracking**: Monitor and handle evaluation failures

## Integration with Evaluation Frameworks

The adapter system integrates seamlessly with evaluation frameworks:

- **Framework Agnostic**: Works with any OpenAI-compatible API
- **Benchmark Independent**: Same interceptors work across different benchmarks
- **Container Compatible**: Integrates with containerized evaluation frameworks

## Next Steps

For detailed implementation information, see:

- **{ref}`nemo-evaluator-interceptors`**: Individual interceptor guides with configuration examples
- **{ref}`interceptor-caching`**: Response caching setup and optimization
- **{ref}`interceptor-reasoning`**: Chain-of-thought processing configuration

The adapter and interceptor system represents a fundamental shift from simple API calls to sophisticated, configurable evaluation workflows that can adapt to diverse research and production needs.
