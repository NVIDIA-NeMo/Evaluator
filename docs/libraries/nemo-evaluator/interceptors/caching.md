# Caching

Cache requests and responses to improve performance and reduce API calls during evaluations.

## Configuration

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.reuse_cached_responses=True'
```

### YAML Configuration
```yaml
interceptors:
  - name: "caching"
    enabled: true
    config:
      cache_dir: "./cache"
      reuse_cached_responses: true
      save_requests: true
      save_responses: true
      max_saved_requests: 1000
      max_saved_responses: 1000
```

## Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `cache_dir` | Directory to store cache files | "./cache" |
| `reuse_cached_responses` | Use cached responses when available | true |
| `save_requests` | Save requests to cache | true |
| `save_responses` | Save responses to cache | true |
| `max_saved_requests` | Maximum number of requests to cache | 1000 |
| `max_saved_responses` | Maximum number of responses to cache | 1000 |

## Endpoint Interceptor

The final interceptor that sends requests to the actual API endpoint.

### YAML Configuration
```yaml
interceptors:
  - name: "endpoint"
    enabled: true
    config: {}
```
