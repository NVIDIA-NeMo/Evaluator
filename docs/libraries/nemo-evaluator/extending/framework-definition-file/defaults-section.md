(defaults-section)=

# Defaults Section

The `defaults` section defines the default configuration and execution command that will be used across all evaluations unless overridden. Overriding is supported either through `--overrides` flag (see {ref}`parameter-overrides`) or {ref}`run-configuration`.

## Command Template

The `command` field uses Jinja2 templating to dynamically generate execution commands based on configuration parameters.

```yaml
defaults:
  command: >-
    {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
    example_eval --model {{target.api_endpoint.model_id}} 
    --task {{config.params.task}}
    --url {{target.api_endpoint.url}} 
    --temperature {{config.params.temperature}}
    # ... additional parameters
```

**Important Note**: `example_eval` is a placeholder representing your actual CLI command. When onboarding your harness, replace this with your real command (e.g., `lm-eval`, `bigcode-eval`, `gorilla-eval`, etc.).

## Template Variables

### Target API Endpoint Variables

- **`{{target.api_endpoint.api_key}}`**: Name of the environment variable storing API key
- **`{{target.api_endpoint.model_id}}`**: Target model identifier
- **`{{target.api_endpoint.stream}}`**: Whether responses should be streamed
- **`{{target.api_endpoint.type}}`**: The type of the target endpoint
- **`{{target.api_endpoint.url}}`**: URL of the model
- **`{{target.api_endpoint.adapter_config}}`**: Adapter configuration

### Evaluation Configuration Variables

- **`{{config.output_dir}}`**: Output directory for results
- **`{{config.type}}`**: Type of the task
- **`{{config.supported_endpoint_types}}`**: Supported endpoint types (chat/completions)

### Configuration Parameters

- **`{{config.params.task}}`**: Evaluation task type
- **`{{config.params.temperature}}`**: Model temperature setting
- **`{{config.params.limit_samples}}`**: Sample limit for evaluation
- **`{{config.params.max_new_tokens}}`**: Maximum tokens to generate
- **`{{config.params.max_retries}}`**: Number of REST request retries
- **`{{config.params.parallelism}}`**: Parallelism to be used
- **`{{config.params.request_timeout}}`**: REST response timeout
- **`{{config.params.top_p}}`**: Top-p sampling parameter
- **`{{config.params.extra}}`**: Framework-specific parameters

## Configuration Defaults

The following example shows common parameter defaults. Each framework defines its own default values in the framework.yml file.

```yaml
defaults:
  config:
    params:
      limit_samples: null           # No limit on samples by default
      max_new_tokens: 4096         # Maximum tokens to generate
      temperature: 0.0             # Deterministic generation
      top_p: 0.00001              # Nucleus sampling parameter
      parallelism: 10              # Number of parallel requests
      max_retries: 5               # Maximum API retry attempts
      request_timeout: 60          # Request timeout in seconds
      extra:                       # Framework-specific parameters
        n_samples: null            # Number of evaluation samples
        downsampling_ratio: null   # Data downsampling ratio
        add_system_prompt: false   # Include system prompt
        args: null                 # Additional CLI arguments
```

## Parameter Categories

### Core Parameters

Basic evaluation settings that control model behavior:
- `temperature`: Controls randomness in generation (0.0 = deterministic)
- `max_new_tokens`: Maximum length of generated output
- `top_p`: Nucleus sampling parameter for diversity

### Performance Parameters

Settings that affect execution speed and reliability:
- `parallelism`: Number of parallel API requests
- `request_timeout`: Maximum wait time for API responses
- `max_retries`: Number of retry attempts for failed requests

### Framework Parameters

Task-specific configuration options:
- `task`: Specific evaluation task to run
- `limit_samples`: Limit number of samples for testing

### Extra Parameters

Custom parameters specific to your framework:
- `n_samples`: Framework-specific sampling configuration
- `downsampling_ratio`: Data subset selection
- `add_system_prompt`: Framework-specific prompt handling
- `args`: Additional CLI arguments passed directly to your framework

## Target Configuration

```yaml
defaults:
  target:
    api_endpoint:
      type: chat                           # Default endpoint type
      supported_endpoint_types:            # All supported types
        - chat
        - completions
        - vlm
        - embedding
```

### Endpoint Types

**chat**: Multi-turn conversation format following the OpenAI chat completions API (`/v1/chat/completions`). Use this for models that support conversational interactions with role-based messages (system, user, assistant).

**completions**: Single-turn text completion format following the OpenAI completions API (`/v1/completions`). Use this for models that generate text based on a single prompt without conversation context. Often used for log-probability evaluations.

**vlm**: Vision-language model endpoints that support image inputs alongside text (`/v1/chat/completions`). Use this for multimodal evaluations that include visual content.

**embedding**: Embedding generation endpoints for retrieval and similarity evaluations (`/v1/embeddings`). Use this for tasks that require vector representations of text.

