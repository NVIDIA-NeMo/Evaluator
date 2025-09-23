# Target Configuration

Target configuration defines the API endpoint for evaluation. Use this section when `deployment: none` is set. You are evaluating an existing endpoint rather than deploying a model.

## Configuration Structure

```yaml
target:
  api_endpoint:
    model_id: your-model-name  # example: meta/llama-3.1-8b-instruct
    url: https://your-endpoint.com/v1/chat/completions  # example: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY  # example: API_KEY
```

## Key Settings

- **`model_id`**: Model identifier.
- **`url`**: Full URL to your OpenAI-compatible endpoint. Use the same URL that you pass to the `curl` command.
- **`api_key_name`**: Name of the environment variable that stores your API key.
  - For NVIDIA APIs, refer to [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key).

## Examples

### NVIDIA API Endpoint

```yaml
target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY
```

### Local Endpoint (No API Key)

```yaml
target:
  api_endpoint:
    model_id: my-local-model
    url: http://localhost:8000/v1/chat/completions
```

## Notes

- For evaluations with deployment, this section is populated automatically.
- The endpoint must be OpenAI-compatible.
- Store API keys as environment variables. Do not hardcode them in configuration files.
