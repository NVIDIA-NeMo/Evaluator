(adapters-client-mode)=

<!-- FIXME: The entire docs/deployment/adapters/ directory is currently orphaned (not linked in the docs navigation).
     This content needs to be reviewed and moved to a proper location
     in the documentation structure. See follow-up PR for reorganization. -->

# Client Mode

The NeMo Evaluator adapter system supports **Client Mode**, where adapters run in-process through a custom httpx transport, providing a simpler alternative to the proxy server architecture.

## Overview

| Feature | Server Mode | Client Mode |
|---------|------------|-------------|
| **Architecture** | Separate proxy server process | In-process via httpx transport |
| **Setup** | Automatic server startup/shutdown | Simple client instantiation |
| **Use Case** | Framework-driven evaluations | Direct API usage, notebooks |
| **Overhead** | Network proxy | Direct in-process execution |
| **Debugging** | Separate process | Same process, easier debugging |

## Quick Start

```python
from nemo_evaluator.client import NeMoEvaluatorClient
from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure model and adapters
config = EndpointModelConfig(
    model_id="my-model",
    url="https://api.example.com/v1/chat/completions",
    api_key_name="API_KEY",  # Environment variable name
    adapter_config=AdapterConfig(
        mode="client",  # Use client mode (no server)
        interceptors=[
            InterceptorConfig(name="caching", enabled=True),
            InterceptorConfig(name="endpoint", enabled=True),
        ]
    ),
    is_base_url=False,  # True if URL is base, False for complete endpoint
)

# Create client
async with NeMoEvaluatorClient(config, output_dir="./output") as client:
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response)
```

## Mode Configuration

### Adapter Mode Field

The `mode` field in `AdapterConfig` controls whether a server process is spawned:

- **`mode="server"`** (default): Spawns adapter server process in `evaluate()` calls
- **`mode="client"`**: Skips server spawning, for use with `NeMoEvaluatorClient`

When using `NeMoEvaluatorClient` directly, set `mode="client"` to prevent unnecessary server creation if the config is also used in `evaluate()` calls.

## URL Modes

Client mode supports two URL configurations via the `is_base_url` flag:

### Base URL Mode (`is_base_url=True`)

Use when the URL is a base URL and the client should append paths:

```python
config = EndpointModelConfig(
    url="https://api.example.com/v1",  # Base URL
    is_base_url=True,
    ...
)
# Requests go to: https://api.example.com/v1/chat/completions
```

### Passthrough Mode (`is_base_url=False`)

Use when the URL is the complete endpoint:

```python
config = EndpointModelConfig(
    url="https://api.example.com/v1/chat/completions",  # Complete endpoint
    is_base_url=False,  # Default
    ...
)
# Requests go to: https://api.example.com/v1/chat/completions (as-is)
```

## API Reference

### Initialization

```python
from nemo_evaluator.client import NeMoEvaluatorClient
from nemo_evaluator.api.api_dataclasses import EndpointModelConfig

client = NeMoEvaluatorClient(
    endpoint_model_config=EndpointModelConfig(
        model_id="model-name",
        url="https://api.example.com/v1/chat/completions",
        api_key_name="API_KEY",
        adapter_config=adapter_config,
        is_base_url=False,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=100,
        request_timeout=60,
        max_retries=3,
        parallelism=5,
    ),
    output_dir="./eval_output"
)
```

### Methods

#### Chat Completion

```python
# Single request (async)
response = await client.chat_completion(
    messages=[{"role": "user", "content": "Hello"}],
    seed=42  # Optional
)

# Batch requests (sync wrapper)
responses = client.chat_completions(
    messages_list=[
        [{"role": "user", "content": "Hello"}],
        [{"role": "user", "content": "Hi"}],
    ],
    seeds=[42, 43],  # Optional
    show_progress=True
)

# Batch requests (async)
responses = await client.batch_chat_completions(
    messages_list=[...],
    seeds=[...],
    show_progress=True
)
```

#### Text Completion

```python
# Single completion
response = await client.completion(
    prompt="Once upon a time",
    seed=42
)

# Batch completions
responses = client.completions(
    prompts=["Prompt 1", "Prompt 2"],
    seeds=[42, 43],
    show_progress=True
)
```

#### Embeddings

```python
# Single embedding
embedding = await client.embedding(text="Hello world")

# Batch embeddings
embeddings = client.embeddings(
    texts=["Text 1", "Text 2"],
    show_progress=True
)
```

### Context Manager

```python
# Recommended: ensures post-eval hooks run
async with NeMoEvaluatorClient(config, output_dir="./output") as client:
    response = await client.chat_completion(messages=[...])
    # Hooks run automatically on exit
```

### Manual Cleanup

```python
client = NeMoEvaluatorClient(config, output_dir="./output")
try:
    response = await client.chat_completion(messages=[...])
finally:
    await client.aclose()  # Runs post-eval hooks
```

## Adapter Configuration

Client mode uses the same `AdapterConfig` as server mode, but with `mode="client"` to prevent server spawning:

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    mode="client",  # Prevents adapter server from spawning
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={"system_message": "You are helpful."}
        ),
        InterceptorConfig(name="request_logging"),
        InterceptorConfig(
            name="caching",
            config={"cache_dir": "./cache"}
        ),
        InterceptorConfig(name="reasoning"),
        InterceptorConfig(name="endpoint"),  # Required
    ],
    post_eval_hooks=[
        {"name": "post_eval_report", "config": {"report_types": ["html"]}}
    ]
)
```

**Note:** When using `NeMoEvaluatorClient`, the `mode` is automatically set to `"client"` if not specified.

## Implementation Details

### Architecture

```
┌─────────────────────────┐
│ Your Script/Notebook    │
└───────────┬─────────────┘
            │
            ↓
┌─────────────────────────┐
│ NeMoEvaluatorClient     │
│ (AsyncOpenAI wrapper)   │
└───────────┬─────────────┘
            │
            ↓
┌─────────────────────────┐
│ AsyncAdapterTransport   │
│ (httpx.AsyncBaseTransport) │
│                         │
│  ┌───────────────────┐  │
│  │ Adapter Pipeline  │  │
│  │ - Interceptors    │  │
│  │ - Post-eval hooks │  │
│  └───────────────────┘  │
└───────────┬─────────────┘
            │ HTTP
            ↓
┌─────────────────────────┐
│ Model Endpoint          │
└─────────────────────────┘
```

### Request Flow

1. User calls `client.chat_completion(...)`
2. AsyncOpenAI client constructs httpx.Request
3. AsyncAdapterTransport intercepts the request
4. Request wrapped for adapter compatibility (HttpxRequestWrapper)
5. Request passes through interceptor chain (in thread pool for sync interceptors)
6. Endpoint interceptor makes HTTP call
7. Response passes back through response interceptors
8. Response converted back to httpx.Response
9. AsyncOpenAI client parses and returns completion

### Sync/Async Bridging

Client mode handles the async/sync boundary automatically:
- AsyncAdapterTransport is async (implements `httpx.AsyncBaseTransport`)
- Adapter pipeline and interceptors are synchronous
- `asyncio.to_thread()` runs sync pipeline in thread pool
- Seamless integration with async OpenAI client

## When to Use Client Mode

### Use Client Mode When:
- Writing custom evaluation scripts
- Working in Jupyter notebooks
- Need direct API control
- Want simpler setup
- Debugging in same process
- Single-process evaluations

### Use Server Mode When:
- Running framework-driven evaluations with `evaluate()`
- Need shared adapter state across processes
- Working with harnesses that don't support custom clients
- Running distributed evaluations

## Examples

### Basic Usage

```python
from nemo_evaluator.client import NeMoEvaluatorClient
from nemo_evaluator.api.api_dataclasses import EndpointModelConfig
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

config = EndpointModelConfig(
    model_id="llama-3-70b",
    url="https://integrate.api.nvidia.com/v1/chat/completions",
    api_key_name="NVIDIA_API_KEY",
    is_base_url=False,
    adapter_config=AdapterConfig(
        interceptors=[
            InterceptorConfig(name="caching"),
            InterceptorConfig(name="endpoint"),
        ]
    ),
)

async with NeMoEvaluatorClient(config, "./output") as client:
    response = await client.chat_completion(
        messages=[{"role": "user", "content": "What is AI?"}]
    )
    print(response)
```

### Batch Processing

```python
# Process multiple prompts with progress bar
prompts = [
    [{"role": "user", "content": f"Question {i}"}]
    for i in range(100)
]

responses = client.chat_completions(
    messages_list=prompts,
    show_progress=True
)
```

### With All Interceptors

```python
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(
            name="system_message",
            config={"system_message": "Be concise."}
        ),
        InterceptorConfig(name="request_logging"),
        InterceptorConfig(name="response_logging"),
        InterceptorConfig(
            name="caching",
            config={
                "cache_dir": "./cache",
                "reuse_cached_responses": True,
                "save_requests": True,
                "save_responses": True,
            }
        ),
        InterceptorConfig(
            name="reasoning",
            config={"start_reasoning_token": "<think>"}
        ),
        InterceptorConfig(name="response_stats"),
        InterceptorConfig(name="endpoint"),
    ],
    post_eval_hooks=[
        {"name": "post_eval_report", "config": {"report_types": ["html", "json"]}}
    ]
)
```

## See Also

- {ref}`adapters-concepts` - Conceptual overview of the adapter system
- {ref}`adapters-configuration` - Available interceptors and configuration options
- {ref}`deployment-adapters-recipes` - Common adapter patterns and recipes
