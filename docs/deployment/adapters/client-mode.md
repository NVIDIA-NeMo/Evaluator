(adapters-client-mode)=

# Client Mode

The NeMo Evaluator adapter system supports two modes of operation: **Server Mode** (the default) and **Client Mode** (new). This page explains both modes and how to choose between them.

## Overview

| Feature | Server Mode | Client Mode |
|---------|------------|-------------|
| **Architecture** | Separate proxy server process | In-process via httpx transport |
| **Setup** | Automatic server startup/shutdown | Simple client instantiation |
| **Use Case** | Framework-driven evaluations | Direct API usage, notebooks |
| **Overhead** | Network proxy | Direct in-process execution |
| **Debugging** | Separate process | Same process, easier debugging |

## Server Mode (Default)

Server mode runs adapters as a separate proxy server that intercepts HTTP requests. This is the default mode when using the `evaluate()` function.

### Example

```python
from nemo_evaluator.api import evaluate
from nemo_evaluator.api.api_dataclasses import (
    Evaluation,
    EvaluationConfig,
    Target,
    APIEndpoint,
    Task,
)
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure adapters
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
            config={"cache_dir": "./cache"}
        ),
        InterceptorConfig(name="reasoning", enabled=True),
        InterceptorConfig(name="endpoint", enabled=True),
    ]
)

# Configure evaluation with adapter
evaluation = Evaluation(
    config=EvaluationConfig(output_dir="./results"),
    target=Target(
        api_endpoint=APIEndpoint(
            url="http://localhost:8000/v1",
            adapter_config=adapter_config  # Server mode
        )
    ),
    tasks=[
        Task(
            name="hellaswag",
            framework="lm-eval",
            task="hellaswag",
            limit=100,
        )
    ]
)

# The adapter server starts automatically
results = evaluate(evaluation)
```

### How Server Mode Works

1. `AdapterServerProcess` starts a Flask server in a separate process
2. The evaluation's target URL is rewritten to point to the local proxy
3. All HTTP requests from the evaluation framework go through the proxy
4. Interceptors process requests and responses in the proxy server
5. Server automatically shuts down after evaluation completes

## Client Mode (New)

Client mode runs adapters in-process through a custom httpx transport. This is ideal when you want direct control over API calls without running a full evaluation.

### Example

```python
from nemo_evaluator.client import NeMoEvaluatorClient
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

# Configure adapters (same as server mode!)
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
            config={"cache_dir": "./cache"}
        ),
        InterceptorConfig(name="reasoning", enabled=True),
        InterceptorConfig(name="endpoint", enabled=True),
    ]
)

# Create OpenAI-compatible client with integrated adapters
with NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    api_key="your-api-key",
    adapter_config=adapter_config,
    output_dir="./eval_output"
) as client:
    # Use like normal OpenAI client - adapters run automatically
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
    print(response.choices[0].message.content)

# Post-eval hooks run automatically on context exit
```

### How Client Mode Works

1. `NeMoEvaluatorClient` extends the OpenAI Python client
2. Custom `AdapterTransport` wraps httpx's base transport
3. Each HTTP request passes through the adapter pipeline in-process
4. Interceptors process requests and responses (same implementations as server mode)
5. Post-eval hooks run when the client is closed

### Usage Without Context Manager

```python
from nemo_evaluator.client import NeMoEvaluatorClient

client = NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    api_key="your-api-key",
    adapter_config=adapter_config,
    output_dir="./output"
)

try:
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello!"}]
    )
finally:
    # Important: close() runs post-eval hooks
    client.close()
```

### Automatic URL Mode Detection

The client automatically detects and adapts to both URL styles at runtime - you don't need to worry about which format to use:

**Base URL Mode (Standard):**
```python
client = NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",  # Base URL
    ...
)
# OpenAI client appends paths: /v1 + /chat/completions
# → Requests go to: http://localhost:8000/v1/chat/completions
```

**Passthrough Mode (Custom Endpoint):**
```python
client = NeMoEvaluatorClient(
    endpoint_url="http://my-server:2137/submit",  # Custom endpoint
    ...
)
# First request tries: http://my-server:2137/submit/chat/completions (OpenAI appends path)
# Gets 404! Automatically retries with: http://my-server:2137/submit (endpoint_url directly)
# → Subsequent requests go to: http://my-server:2137/submit
```

**How Detection Works:**

1. On the first request, the client tries the URL as constructed by the OpenAI client  
   (endpoint_url + endpoint path like `/chat/completions`)
2. If the server returns 404 (Not Found) or 405 (Method Not Allowed)
3. The client automatically retries using the `endpoint_url` directly (passthrough mode)
4. Remembers which mode worked for all subsequent requests (no more retries)

This adaptive approach works with:
- Standard OpenAI-compatible APIs
- Custom endpoints at any URL path
- Any server configuration


## When to Use Each Mode

### Use Server Mode When:

- Running framework-driven evaluations with `evaluate()`
- Need to share adapter state across multiple evaluation processes
- Working with evaluation harnesses that don't support custom clients
- Running distributed evaluations across multiple processes
- Need to intercept requests from tools you don't control

### Use Client Mode When:

- Writing custom evaluation scripts with direct API calls
- Working in Jupyter notebooks or interactive environments
- Need simpler setup without managing server processes
- Want easier debugging (same process, clearer stack traces)
- Building custom evaluation workflows with OpenAI client
- Running single-process evaluations

## Configuration Compatibility

Both modes use the **exact same** `AdapterConfig` and interceptor implementations. You can switch between modes without changing your adapter configuration:

```python
# This config works in BOTH modes
adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(name="system_message", enabled=True, 
                         config={"system_message": "Be concise."}),
        InterceptorConfig(name="caching", enabled=True),
        InterceptorConfig(name="endpoint", enabled=True),
    ]
)

# Server mode - use with evaluate()
evaluation = Evaluation(
    target=Target(
        api_endpoint=APIEndpoint(
            url="http://localhost:8000/v1",
            adapter_config=adapter_config  # ← Same config
        )
    ),
    # ...
)

# Client mode - use with NeMoEvaluatorClient
client = NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    adapter_config=adapter_config,  # ← Same config
    output_dir="./output"
)
```

## Advanced Client Mode Usage

### Custom HTTP Client

Provide a custom httpx client to be wrapped with adapter transport:

```python
import httpx

custom_http_client = httpx.Client(
    timeout=httpx.Timeout(60.0),
    limits=httpx.Limits(max_keepalive_connections=5)
)

client = NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    api_key="your-api-key",
    adapter_config=adapter_config,
    http_client=custom_http_client  # Will be wrapped with adapters
)
```

### Async Operations

Client mode supports async operations with the OpenAI client:

```python
import asyncio
from nemo_evaluator.client import NeMoEvaluatorClient

async def run_evaluation():
    async with NeMoEvaluatorClient(
        endpoint_url="http://localhost:8000/v1",
        api_key="your-api-key",
        adapter_config=adapter_config,
        output_dir="./output"
    ) as client:
        response = await client.chat.completions.create(
            model="my-model",
            messages=[{"role": "user", "content": "Hello!"}]
        )
        return response

asyncio.run(run_evaluation())
```

### Manual Post-Eval Hook Execution

Control when post-eval hooks are executed:

```python
client = NeMoEvaluatorClient(...)

# Do work...
for prompt in prompts:
    response = client.chat.completions.create(...)
    process_response(response)

# Manually run post-eval hooks when done
client.close()
```

## Implementation Details

### Architecture Diagram

```
Server Mode:                      Client Mode:
┌─────────────┐                  ┌─────────────┐
│ Eval Harness│                  │ Your Script │
└──────┬──────┘                  └──────┬──────┘
       │ HTTP                           │
       ↓                                ↓
┌──────────────┐              ┌──────────────────┐
│ Adapter      │              │ OpenAI Client    │
│ Server       │              │ ┌──────────────┐ │
│ (Proxy)      │              │ │ Adapter      │ │
│              │              │ │ Transport    │ │
│ Interceptors │              │ │              │ │
└──────┬───────┘              │ │ Interceptors │ │
       │ HTTP                 │ └──────────────┘ │
       ↓                      └────────┬─────────┘
┌──────────────┐                      │ HTTP
│ Model        │                      ↓
│ Endpoint     │              ┌──────────────┐
└──────────────┘              │ Model        │
                              │ Endpoint     │
                              └──────────────┘
```

### Request Flow in Client Mode

1. User calls `client.chat.completions.create(...)`
2. OpenAI client constructs `httpx.Request`
3. `AdapterTransport.handle_request()` intercepts the request
4. Request passes through interceptor chain:
   - System message interceptor adds/modifies system prompt
   - Request logging interceptor logs the request
   - Caching interceptor checks cache (may return cached response)
   - Endpoint interceptor makes actual HTTP call
5. Response passes back through response interceptors:
   - Reasoning interceptor extracts reasoning tokens
   - Response logging interceptor logs response
   - Response stats interceptor collects metrics
6. Response converted to `httpx.Response`
7. OpenAI client receives and parses response
8. User gets completion object

### Type Conversions

The adapter pipeline uses Flask/requests types, but client mode uses httpx. The implementation provides transparent conversion:

- **Request Phase**: `httpx.Request` → `HttpxRequestWrapper` (Flask-like interface) → `AdapterRequest`
- **Response Phase**: `AdapterResponse` → `RequestsResponseWrapper` (requests-like interface) → `httpx.Response`

These wrapper classes provide interface compatibility without modifying existing interceptor implementations.

## Migration Guide

### From Direct OpenAI Client to Client Mode

**Before:**
```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

response = client.chat.completions.create(
    model="my-model",
    messages=[{"role": "user", "content": "Hello"}]
)
```

**After (with adapters):**
```python
from nemo_evaluator.client import NeMoEvaluatorClient
from nemo_evaluator.adapters.adapter_config import AdapterConfig, InterceptorConfig

adapter_config = AdapterConfig(
    interceptors=[
        InterceptorConfig(name="request_logging", enabled=True),
        InterceptorConfig(name="caching", enabled=True),
        InterceptorConfig(name="endpoint", enabled=True),
    ]
)

with NeMoEvaluatorClient(
    endpoint_url="http://localhost:8000/v1",
    api_key="your-api-key",
    adapter_config=adapter_config,
    output_dir="./output"
) as client:
    response = client.chat.completions.create(
        model="my-model",
        messages=[{"role": "user", "content": "Hello"}]
    )
```

The API is identical to OpenAI client - just add adapter configuration!

## See Also

- {ref}`adapters-concepts` - Conceptual overview of the adapter system
- {ref}`adapters-configuration` - Available interceptors and configuration options
- {ref}`deployment-adapters-recipes` - Common adapter patterns and recipes

