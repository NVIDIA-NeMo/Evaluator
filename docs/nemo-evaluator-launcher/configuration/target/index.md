# Target Configuration

Target configuration defines the API endpoint to evaluate. This section is used when `deployment: none` is specified, meaning you're using an existing endpoint rather than deploying your own model.

## Configuration Structure

```yaml
target:
  api_endpoint:
    model_id: your-model-name  # example: meta/llama-3.1-8b-instruct
    url: https://your-endpoint.com/v1/chat/completions  # example: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: API_KEY  # example: API_KEY
```

## Key Settings

- **`model_id`**: Name/identifier of your model
- **`url`**: Full URL to your OpenAI-compatible endpoint (exactly the same URL you would use in a bash curl request)
- **`api_key_name`**: Environment variable name containing your API key
  - For NVIDIA APIs, see [Setting up API Keys](https://docs.omniverse.nvidia.com/guide-sdg/latest/setup.html#preview-and-set-up-an-api-key)

## Examples

# NVIDIA Build Endpoint


```yaml
target:
  api_endpoint:
    model_id: meta/llama-3.1-8b-instruct
    url: https://integrate.api.nvidia.com/v1/chat/completions
    api_key_name: NGC_API_KEY
```

# Local Endpoint (API KEY not needed)
```yaml
target:
  api_endpoint:
    model_id: my-local-model
    url: http://localhost:8000/v1/chat/completions
```

## Notes

- For evaluations with deployment, this section is automatically populated
- The endpoint must be OpenAI-compatible
- API keys should be stored as environment variables, not hardcoded in configuration files
