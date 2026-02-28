(interceptor-caching)=

# Caching

## Overview

The `CachingInterceptor` implements a caching system that can store responses based on request content, enabling faster re-runs of evaluations and reducing costs when using paid APIs.

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_caching=True,target.api_endpoint.adapter_config.caching_dir=./cache,target.api_endpoint.adapter_config.reuse_cached_responses=True'
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

For detailed configuration options, please refer to the {ref}`interceptor_reference` Python API reference.

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

- **Response Cache** (`{cache_dir}/responses/`): Stores raw response content (bytes) keyed by cache key (when `save_responses=True` or `reuse_cached_responses=True`)
- **Headers Cache** (`{cache_dir}/headers/`): Stores response headers (dictionary) keyed by cache key (when `save_requests=True`)
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

## Seed Cache

Use `seed_cache_dir` to reuse cached responses from a previous run — e.g., when resuming after a timeout or migrating between clusters.

```yaml
target:
  api_endpoint:
    adapter_config:
      use_caching: true
      seed_cache_dir: /path/to/previous/run/cache
```

On a primary cache miss, the interceptor falls back to the seed cache. Seed hits are **automatically promoted** into the primary cache, so the output cache is self-contained. The seed cache is never modified.

Cache keys are SHA-256 hashes of the request JSON body — portable across clusters as long as the request data is identical.
