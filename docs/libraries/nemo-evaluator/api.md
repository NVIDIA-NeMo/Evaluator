# API Reference



## Available Data Classes

The API provides several dataclasses for configuration:

```python
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig,      # Main evaluation configuration
    EvaluationTarget,      # Target model configuration
    ConfigParams,          # Evaluation parameters
    ApiEndpoint,           # API endpoint configuration
    EvaluationResult,      # Evaluation results
    TaskResult,            # Individual task results
    MetricResult,          # Metric scores
    Score,                 # Score representation
    ScoreStats,            # Score statistics
    GroupResult,           # Grouped results
    EndpointType,          # Endpoint type enum
    Evaluation             # Complete evaluation object
)
```

## `run_eval`

The main entry point for running evaluations. This is a CLI entry point that parses command line arguments.

```python
from nemo_evaluator.api.run import run_eval

def run_eval() -> None:
    """
    CLI entry point for running evaluations.
    
    This function parses command line arguments and executes evaluations.
    It does not take parameters directly - all configuration is passed through CLI arguments.
    
    CLI Arguments:
        --eval_type: Type of evaluation to run (such as "mmlu_pro", "gpqa_diamond")
        --model_id: Model identifier (such as "meta/llama-3.1-8b-instruct")
        --model_url: API endpoint URL (such as "https://integrate.api.NVIDIA.com/v1/chat/completions" for chat endpoint type)
        --model_type: Endpoint type ("chat", "completions", "vlm", "embedding")
        --api_key_name: Environment variable name for API key integration with endpoints (optional)
        --output_dir: Output directory for results
        --run_config: Path to YAML Run Configuration file (optional)
        --overrides: Comma-separated dot-style parameter overrides (optional)
        --dry_run: Show rendered config without running (optional)
        --debug: Enable debug logging (optional, deprecated, use NV_LOG_LEVEL=DEBUG env var)
    
    Usage:
        run_eval()  # Parses sys.argv automatically
    """
```

:::{note} The `run_eval()` function is designed as a CLI entry point. For programmatic usage, use the underlying configuration objects and the `evaluate()` function directly.
:::

## `evaluate`

The core evaluation function for programmatic usage.

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, EvaluationTarget

def evaluate(
    eval_cfg: EvaluationConfig,
    target_cfg: EvaluationTarget
) -> EvaluationResult:
    """
    Run an evaluation using configuration objects.
    
    Args:
        eval_cfg: Evaluation configuration object
        target_cfg: Target configuration object
    
    Returns:
        EvaluationResult: Evaluation results and metadata
    """
```



**Example Programmatic Usage:**

```python
from nemo_evaluator.core.evaluate import evaluate
from nemo_evaluator.api.api_dataclasses import (
    EvaluationConfig, 
    EvaluationTarget, 
    ConfigParams,
    ApiEndpoint
)

# Create evaluation configuration
eval_config = EvaluationConfig(
    type="simple_evals.mmlu_pro",
    output_dir="./results", 
    params=ConfigParams(
        limit_samples=100,
        temperature=0.1
    )
)

# Create target configuration
target_config = EvaluationTarget(
    api_endpoint=ApiEndpoint(
        url="https://integrate.api.NVIDIA.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        type="chat",
        api_key="MY_API_KEY" # Name of the environment variable that stores api_key
    )
)

# Run evaluation
result = evaluate(eval_config, target_config)
```

## Data Structures

### `EvaluationConfig`

Configuration for evaluation runs, defined in `api_dataclasses.py`.

```python
from nemo_evaluator.api.api_dataclasses import EvaluationConfig

class EvaluationConfig:
    """Configuration for evaluation runs."""
    type: str                    # Type of evaluation - benchmark to be run
    output_dir: str              # Output directory
    params: str                  # parameter overrides
```

### `EvaluationTarget`

Target configuration for API endpoints, defined in `api_dataclasses.py`.

```python
from nemo_evaluator.api.api_dataclasses import EvaluationTarget, EndpointType

class EvaluationTarget:
    """Target configuration for API endpoints."""
    api_endpoint: ApiEndpoint  # API endpoint to be used for evaluation
 
class ApiEndpoint:
    url: str                          # API endpoint URL
    model_id: str                     # Model name or identifier
    type: str                         # Endpoint type (chat, completions, vlm, or embedding)
    api_key: str                      # Name of the env variable that stores API key
    adapter_config: AdapterConfig     # Adapter configuration
```

In the ApiEndpoint dataclass, `type` should be one of: `EndpointType.CHAT`, `EndpointType.COMPLETIONS`, `EndpointType.VLM`, `EndpointType.EMBEDDING`:
    - `CHAT` endpoint accepts structured input as a sequence of messages (such as system, user, assistant roles). It returns a model-generated message, enabling controlled multi-turn interactions.
    - `COMPLETIONS` endpoint takes a single prompt string and returns a text continuation, typically used for one-shot or single-turn tasks without conversational structure.
    - `VLM` endpoint hosts a model that has vision capabilities.
    - `EMBEDDING` endpoint hosts an embedding model.

## Adapter System

### `AdapterConfig`

Configuration for the adapter system, defined in `adapter_config.py`.

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

class AdapterConfig:
    """Configuration for the adapter system."""
    
    discovery: DiscoveryConfig                    # Module discovery configuration
    interceptors: list[InterceptorConfig]        # List of interceptors
    post_eval_hooks: list[PostEvalHookConfig]   # Post-evaluation hooks
    endpoint_type: str                           # Default endpoint type
    caching_dir: str | None                      # Legacy caching directory
```

### `InterceptorConfig`

Configuration for individual interceptors.

```python
from nemo_evaluator.adapters.adapter_config import InterceptorConfig

class InterceptorConfig:
    """Configuration for a single interceptor."""
    
    name: str                        # Interceptor name
    enabled: bool                    # Whether enabled
    config: dict[str, Any]          # Interceptor-specific configuration
```

### `DiscoveryConfig`

Configuration for discovering third-party modules and directories.

```python
from nemo_evaluator.adapters.adapter_config import DiscoveryConfig

class DiscoveryConfig:
    """Configuration for discovering 3rd party modules and directories."""
    
    modules: list[str]               # List of module paths to discover
    dirs: list[str]                  # List of directory paths to discover
```

## Available Interceptors

### 1. Request Logging Interceptor

```python
from nemo_evaluator.adapters.interceptors.logging_interceptor import LoggingInterceptor

# Configuration
interceptor_config = {
    "name": "request_logging",
    "enabled": True,
    "config": {
        "output_dir": "/tmp/logs",
        "max_requests": 1000,
        "log_failed_requests": True
    }
}
```

**Features:**

- Logs all API requests and responses
- Configurable output directory
- Request/response count limits
- Failed request logging

### 2. Caching Interceptor

```python
from nemo_evaluator.adapters.interceptors.caching_interceptor import CachingInterceptor

# Configuration
interceptor_config = {
    "name": "caching",
    "enabled": True,
    "config": {
        "cache_dir": "/tmp/cache",
        "reuse_cached_responses": True,
        "save_requests": True,
        "save_responses": True,
        "max_saved_requests": 1000,
        "max_saved_responses": 1000
    }
}
```

**Features:**

- Response caching for performance
- Persistent storage - responses are saved to disk, allowing resumption after process termination
- Configurable cache directory
- Request/response persistence
- Cache size limits

### 3. Reasoning Interceptor

```python
from nemo_evaluator.adapters.interceptors.reasoning_interceptor import ReasoningInterceptor

# Configuration
interceptor_config = {
    "name": "reasoning",
    "enabled": True,
    "config": {
        "start_reasoning_token": "<think>",
        "end_reasoning_token": "</think>",
        "add_reasoning": True,
        "enable_reasoning_tracking": True
    }
}
```

**Features:**

- Reasoning chain support
- Custom reasoning tokens
- Reasoning tracking and analysis
- Chain-of-thought prompting

### 4. System Message Interceptor

```python
from nemo_evaluator.adapters.interceptors.system_message_interceptor import SystemMessageInterceptor

# Configuration
interceptor_config = {
    "name": "system_message",
    "enabled": True,
    "config": {
        "system_message": "You are a helpful AI assistant.",
        "strategy": "prepend"  # Optional: "replace", "append", or "prepend" (default)
    }
}
```

**Features:**

- Custom system prompt injection
- Multiple strategies for handling existing system messages (replace, prepend, append)
- Consistent system behavior
- Flexible system message composition

**Use Cases:**

- Modify system prompts for different evaluation scenarios
- Test different prompt variations without code changes
- Replace existing system messages for consistent behavior
- Prepend or append instructions to existing system messages
- A/B testing of different prompt strategies

### 5. Endpoint Interceptor

```python
from nemo_evaluator.adapters.interceptors.endpoint_interceptor import EndpointInterceptor

# Configuration
interceptor_config = {
    "name": "endpoint",
    "enabled": True,
    "config": {
        "endpoint_url": "https://api.example.com/v1/chat/completions",
        "timeout": 30
    }
}
```

**Features:**

- Endpoint URL management
- Request timeout configuration
- Endpoint validation

### 6. Payload Modifier Interceptor

```python
from nemo_evaluator.adapters.interceptors.payload_modifier_interceptor import PayloadModifierInterceptor

# Configuration
interceptor_config = {
    "name": "payload_modifier",
    "enabled": True,
    "config": {
        "params_to_add": {
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False
                }
            }
        },
        "params_to_remove": ["field_in_msgs_to_remove"],
        "params_to_rename": {"max_tokens": "max_completion_tokens"}
    }
}
```

**Explanation:**

This interceptor is particularly useful when custom behavior is needed. In this example, the `enable_thinking` parameter is a custom key that controls the reasoning mode of the model. When set to `False`, it disables the model's internal reasoning/thinking process, which can be useful for scenarios where you want more direct responses without the model's step-by-step reasoning output.  
The `field_in_msgs_to_remove` field would be removed recursively from all messages in the payload.

**Features:**

- Custom parameter injection
- Remove fields recursively at all levels of the payload
- Rename top-level payload keys

### 7. Client Error Interceptor

```python
from nemo_evaluator.adapters.interceptors.raise_client_error_interceptor import RaiseClientErrorInterceptor

# Configuration
interceptor_config = {
    "name": "raise_client_error",
    "enabled": True,
    "config": {
        "raise_on_error": True,
        "error_threshold": 400
    }
}
```

**Features:**

- Error handling and propagation
- Configurable error thresholds
- Client error management
