(adapters-configuration)=

# Configuration

The adapter system is configured using the `AdapterConfig` class with the following options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `api_url` | `str` |  | URL of the api endpoint (same as `target_config.api_endpoint.url`) that will be proxied |
| `local_port` | `Optional[int]` | `None` | Local port to use for the adapter server. If `None`, a free port will be chosen automatically |
| `use_reasoning` | `bool` | `False` | Whether to use the clean-reasoning-tokens adapter |
| `end_reasoning_token` | `str` | `"</think>"` | Token that signifies the end of reasoning output |
| `custom_system_prompt` | `Optional[str]` | `None` | A custom system prompt to replace the original one (if not None), **only for chat endpoints** |
| `max_logged_responses` | `Optional[int]` | `5` | Maximum number of responses to log. Set to 0 to disable. If None, all will be logged |
| `max_logged_requests` | `Optional[int]` | `5` | Maximum number of requests to log. Set to 0 to disable. If None, all will be logged |


