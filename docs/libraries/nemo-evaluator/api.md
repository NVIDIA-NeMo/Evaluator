(nemo-evaluator-api)=

# API Reference

Access the complete NeMo Evaluator Python API through this comprehensive reference guide.

## Core API Functions

Choose from multiple API layers based on your needs:

### API Layers

1. **Core Evaluation API** (`nemo_evaluator.core.evaluate`): Direct evaluation with full adapter support
2. **High-level API** (`nemo_evaluator.api.run`): Simplified interface for common workflows  
3. **CLI Interface** (`nemo_evaluator.cli`): Command-line evaluation tools

### When to Use Each Layer

- **Core API**: Maximum flexibility, custom interceptors, integration into ML pipelines
- **High-level API**: Standard evaluations with adapter configuration
- **CLI**: Quick evaluations, scripting, and automation

### Available Dataclasses

Configure your evaluations using these dataclasses:

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

## Core Evaluation API

### `run_eval`

CLI entry point for running evaluations. This function parses command line arguments.

```python
from nemo_evaluator.api.run import run_eval

def run_eval() -> None:
    """
    CLI entry point for running evaluations.
    
    This function parses command line arguments and executes evaluations.
    It does not take parameters directly - all configuration is passed via CLI arguments.
    
    CLI Arguments:
        --eval_type: Type of evaluation to run (e.g., "mmlu_pro", "gsm8k")
        --model_id: Model identifier (e.g "meta/llama-3.1-8b-instruct")
        --model_url: API endpoint URL (e.g "https://integrate.api.nvidia.com/v1/chat/completions" for chat endpoint type)
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

**Note**: The `run_eval()` function is designed as a CLI entry point. For programmatic usage, you should use the `evaluate()` function directly with configuration objects.

### `evaluate`

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
        eval_cfg: Evaluation configuration object containing output directory,
                  parameters, and evaluation type
        target_cfg: Target configuration object containing API endpoint details
                    and adapter configuration
    
    Returns:
        EvaluationResult: Evaluation results and metadata
    """
```

**Prerequisites:**
- **Container way**: Use simple-evals container mentioned in {ref}`nemo-evaluator-containers`
- **Python way**: 
  ```bash
  pip install nemo-evaluator nvidia-simple-evals
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
        url="https://integrate.api.nvidia.com/v1/chat/completions",
        model_id="meta/llama-3.1-8b-instruct",
        type="chat",
        api_key="your_api_key_here"
    )
)

# Run evaluation
result = evaluate(eval_config, target_config)
```

## Data Structures

### `EvaluationConfig`

Configuration for evaluation runs, defined in `api_dataclasses.py`.

```python
from nemo_evaluator.api.api_dataclasses import EvaluationConfig, ConfigParams

class EvaluationConfig:
    """Configuration for evaluation runs."""
    output_dir: Optional[str]                      # Directory to output results
    params: Optional[ConfigParams]                 # Evaluation parameters
    supported_endpoint_types: Optional[list[str]]  # Supported endpoint types
    type: Optional[str]                            # Type of evaluation task
```

### `ConfigParams`

Parameters for evaluation execution.

```python
from nemo_evaluator.api.api_dataclasses import ConfigParams

class ConfigParams:
    """Parameters for evaluation execution."""
    limit_samples: Optional[int | float]  # Limit number of evaluation samples
    max_new_tokens: Optional[int]         # Maximum tokens to generate
    max_retries: Optional[int]            # Number of REST request retries
    parallelism: Optional[int]            # Parallelism level
    task: Optional[str]                   # Name of the task
    temperature: Optional[float]          # Sampling temperature (0.0-1.0)
    request_timeout: Optional[int]        # REST response timeout
    top_p: Optional[float]                # Top-p sampling parameter (0.0-1.0)
    extra: Optional[Dict[str, Any]]       # Framework-specific parameters
```

### `EvaluationTarget`

Target configuration for API endpoints, defined in `api_dataclasses.py`.

```python
from nemo_evaluator.api.api_dataclasses import EvaluationTarget, ApiEndpoint

class EvaluationTarget:
    """Target configuration for API endpoints."""
    api_endpoint: Optional[ApiEndpoint]  # API endpoint configuration

class ApiEndpoint:
    """API endpoint configuration."""
    api_key: Optional[str]                      # API key or env variable name
    model_id: Optional[str]                     # Model identifier
    stream: Optional[bool]                      # Whether to stream responses
    type: Optional[EndpointType]                # Endpoint type (chat, completions, vlm, embedding)
    url: Optional[str]                          # API endpoint URL
    adapter_config: Optional[AdapterConfig]     # Adapter configuration
```

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
        "custom_system_prompt": "You are a helpful AI assistant.",
        "override_existing": True
    }
}
```

**Features:**
- Custom system prompt injection
- Prompt override capabilities
- Consistent system behavior

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

### 6. Progress Tracking Interceptor

```python
from nemo_evaluator.adapters.interceptors.progress_tracking_interceptor import ProgressTrackingInterceptor

# Configuration
interceptor_config = {
    "name": "progress_tracking",
    "enabled": True,
    "config": {
        "tracking_url": "http://localhost:3828/progress",
        "interval": 1,
        "track_requests": True,
        "track_responses": True
    }
}
```

**Features:**
- Real-time progress monitoring
- Configurable tracking intervals
- Request/response tracking
- External progress endpoints

### 7. Payload Modifier Interceptor

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
        }
    }
}
```

**Features:**
- Request payload modification
- Custom parameter injection
- Flexible configuration options

### 8. Client Error Interceptor

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

## Configuration Examples

### Basic Framework Configuration

```yaml
framework:
  name: mmlu_pro
  defaults:
    config:
      params:
        limit_samples: 100
        max_tokens: 512
        temperature: 0.1
    target:
      api_endpoint:
        adapter_config:
          interceptors:
            - name: "request_logging"
              enabled: true
              config:
                output_dir: "./logs"
            - name: "caching"
              enabled: true
              config:
                cache_dir: "./cache"
```

### Advanced Adapter Configuration

```yaml
framework:
  name: advanced_eval
  defaults:
    target:
      api_endpoint:
        adapter_config:
          discovery:
            modules: ["custom.interceptors", "my.package"]
            dirs: ["/path/to/custom/interceptors"]
          interceptors:
            - name: "request_logging"
              enabled: true
              config:
                output_dir: "./logs"
                max_requests: 1000
            - name: "caching"
              enabled: true
              config:
                cache_dir: "./cache"
                reuse_cached_responses: true
            - name: "reasoning"
              enabled: true
              config:
                start_reasoning_token: "<think>"
                end_reasoning_token: "</think>"
                add_reasoning: true
            - name: "progress_tracking"
              enabled: true
              config:
                tracking_url: "http://localhost:3828/progress"
                interval: 1
          post_eval_hooks:
            - name: "custom_analysis"
              enabled: true
              config:
                analysis_type: "detailed"
          endpoint_type: "chat"
          caching_dir: "./legacy_cache"
```

## Interceptor System

The NeMo Evaluator uses an interceptor-based architecture that processes requests and responses through a configurable chain of components. Interceptors can modify requests, responses, or both, and can be enabled/disabled and configured independently.

### Configuration Methods

There are two primary ways to configure interceptors:

1. **CLI Overrides**: Use the `--overrides` parameter for runtime configuration
2. **YAML Configuration**: Define interceptor chains in configuration files

### Configure Interceptors

Refer to {ref}`nemo-evaluator-interceptors` for details.

### Complete Configuration Example

Here's a complete example combining multiple interceptors:

**CLI Configuration:**
```bash
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --overrides 'target.api_endpoint.adapter_config.use_request_logging=True,target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.generate_html_report=True'
```

**YAML Configuration:**
```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "request_logging"
          enabled: true
          config:
            output_dir: "./logs"
            max_requests: 1000
            log_failed_requests: true
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./cache"
            reuse_cached_responses: true
            save_requests: true
            save_responses: true
        - name: "endpoint"
          enabled: true
        - name: "response_logging"
          enabled: true
          config:
            output_dir: "./logs"
            max_responses: 1000
      post_eval_hooks:
        - name: "post_eval_report"
          enabled: true
          config:
            report_types: ["html", "json"]
```
To use the above, save it as `config.yaml` and run:
```
eval-factory run_eval \
    --eval_type mmlu_pro \
    --model_id meta/llama-3.1-8b-instruct \
    --model_url https://integrate.api.nvidia.com/v1/chat/completions \
    --model_type chat \
    --api_key_name MY_API_KEY \
    --output_dir ./results \
    --run_config config.yaml
```

### Interceptor Chain Order

Interceptors are executed in the order they appear in the configuration. A typical order is:

1. `system_message` (add/modify system prompts)
2. `payload_modifier` (modify request parameters)
3. `omni_info` (add omni info if specified)
4. `request_logging` (capture final request)
5. `caching` (check cache before making request)
6. `endpoint` or `nvcf` (make the actual API call)
7. `response_logging` (log the response)
8. `reasoning` (add/process reasoning tokens)
9. `progress_tracking` (track evaluation progress)

If the interceptors are configured from the CLI, the chain will be configured in the above order.
