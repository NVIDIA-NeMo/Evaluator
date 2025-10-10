(interceptor-caching)=

# Caching

## Overview

The `CachingInterceptor` implements aREHRJ sophisticated caching system that can store responses based on request content, enabling faster re-runs of evaluations and reducing costs when using paid APIs.

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.interceptors=[{"name":"caching","enabled":true,"config":{"cache_dir":"./cache","reuse_cached_responses":true}}]'
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./evaluation_cache"
            reuse_cached_responses: true
            save_requests: true
            save_responses: true
            max_saved_requests: 1000
            max_saved_responses: 1000
        - name: "endpoint"
          enabled: true
          config: {}
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `cache_dir` | Directory to store cache files | `"/tmp"` | str |
| `reuse_cached_responses` | Use cached responses when available | `False` | bool |
| `save_requests` | Save requests to cache storage | `False` | bool |
| `save_responses` | Save responses to cache storage | `True` | bool |
| `max_saved_requests` | Maximum number of requests to save | `None` | int \| None |
| `max_saved_responses` | Maximum number of responses to cache | `None` | int \| None |

## Cache Key Generation

The interceptor generates the cache key by creating a SHA256 hash of the JSON-serialized request data using `json.dumps()` with `sort_keys=True` for consistent ordering.

```python
import hashlib
import json

# Request data
request_data = {
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "temperature": 0.0,
    "max_new_tokens": 512
}

# Generate cache key
data_str = json.dumps(request_data, sort_keys=True)
cache_key = hashlib.sha256(data_str.encode("utf-8")).hexdigest()
# Result: "abc123def456..." (64-character hex string)
```

## Cache Storage Format

The caching interceptor stores data in three separate disk-backed key-value stores within the configured cache directory:

- **Response Cache** (`{cache_dir}/responses/`): Stores raw response content (bytes) keyed by cache key
- **Headers Cache** (`{cache_dir}/headers/`): Stores response headers (dictionary) keyed by cache key  
- **Request Cache** (`{cache_dir}/requests/`): Stores request data (dictionary) keyed by cache key (when `save_requests=True`)

Each cache uses a SHA256 hash of the request data as the lookup key. When a cache hit occurs, the interceptor retrieves both the response content and headers using the same cache key.

## Cache Behavior

### Cache Hit Process

1. **Request arrives** at the caching interceptor
2. **Generate cache key** from request parameters  
3. **Check cache** for existing response
4. **Return cached response** if found (sets `cache_hit=True`)
5. **Skip API call** and continue to next interceptor

### Cache Miss Process

1. **Request continues** to endpoint interceptor
2. **Response received** from model API
3. **Store response** in cache with generated key
4. **Continue processing** with response interceptors
