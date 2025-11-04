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

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `cache_dir` | Directory to store local cache files (3 separate directories) | `"/tmp"` | str |
| `prefill_from_export` | Path to exported .cache file to prefill cache from | `None` | str |
| `export_cache` | Export cache to single .cache file (pickle format) | `False` | bool |
| `reuse_cached_responses` | Use cached responses when available | `False` | bool |
| `save_requests` | Save requests to cache storage | `False` | bool |
| `save_responses` | Save responses to cache storage | `True` | bool |
| `max_saved_requests` | Maximum number of requests to save | `None` | int \| None |
| `max_saved_responses` | Maximum number of responses to cache | `None` | int \| None |
| `test_mode` | Fail on cache miss with diff to most similar request | `False` | bool |

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

## Cache Export and Import

The caching interceptor supports exporting the entire cache to a single portable binary file (`.cache` format) and importing it later. This is useful for:

- **Sharing caches** between team members or machines
- **Version control** of evaluation caches
- **CI/CD pipelines** where you want deterministic evaluation results
- **Offline evaluation** using pre-generated responses

### Exporting Cache

To export the cache after evaluation, enable the `export_cache` parameter:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./evaluation_cache"
            save_requests: true
            save_responses: true
            export_cache: true  # Export to cache_export.cache
```

This creates a `cache_export.cache` file in the output directory containing all cached requests, responses, and headers in a single pickled file.

### Importing Cache

To prefill the cache from a previously exported file:

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./local_cache"
            prefill_from_export: "/path/to/cache_export.cache"
            reuse_cached_responses: true
```

The cache will be loaded at evaluation start, and all matching requests will use the cached responses.

### Cache Format Comparison

**Local Cache** (`cache_dir`):
- Three separate directories: `requests/`, `responses/`, `headers/`
- Uses disk-backed SQLite databases
- Efficient for ongoing evaluations
- Not easily portable

**Export Cache** (`export_cache`/`prefill_from_export`):
- Single binary `.cache` file (pickle format)
- Portable and shareable
- Version control friendly
- Can be used across machines

## Test Mode

Test mode is a debugging feature that helps identify when requests have changed between evaluation runs. When enabled with cached responses, it will fail on the first cache miss and show a diff with the most similar cached request.

### Enabling Test Mode

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "caching"
          enabled: true
          config:
            cache_dir: "./evaluation_cache"
            prefill_from_export: "/path/to/baseline_cache.cache"
            reuse_cached_responses: true
            test_mode: true  # Fail on cache miss
```

### Use Cases

**1. Regression Testing**
```bash
# First run: Create baseline cache
nemo-evaluator run --config eval.yaml --overrides 'target.api_endpoint.adapter_config.interceptors[0].config.export_cache=true'

# Later: Test with baseline cache
nemo-evaluator run --config eval.yaml --overrides 'target.api_endpoint.adapter_config.interceptors[0].config.prefill_from_export=cache_export.cache,target.api_endpoint.adapter_config.interceptors[0].config.test_mode=true'
```

**2. Debugging Request Changes**

When test mode detects a cache miss, it will:
1. Find the most similar cached request using fuzzy string matching (rapidfuzz)
2. Calculate similarity score (0-100%)
3. Generate a unified diff showing the differences
4. Raise `CacheMissInTestModeError` with detailed information

Example error output:
```
CacheMissInTestModeError: Cache miss in test mode for request with cache_key=abc123def456...

Most similar cached request (similarity: 94.56%):
--- cached_request
+++ current_request
@@ -1,7 +1,7 @@
 {
   "messages": [
     {
       "content": "What is 2+2?",
       "role": "user"
     }
   ],
-  "temperature": 0.0
+  "temperature": 0.7
 }
```

This helps you quickly identify what changed in your request format or parameters.

### Exception Handling

```python
from nemo_evaluator.adapters.interceptors.caching_interceptor import CacheMissInTestModeError

try:
    # Run evaluation with test_mode=True
    evaluator.evaluate(dataset)
except CacheMissInTestModeError as e:
    print(f"Cache miss for request: {e.request_data}")
    if e.most_similar_request:
        print(f"Most similar (score: {e.similarity_score:.2f}%)")
        print(f"Diff:\n{e.diff}")
    else:
        print("No similar cached requests found")
```

## Best Practices

### Development Workflow

1. **Initial Run**: Create a baseline cache
   ```yaml
   save_requests: true
   save_responses: true
   export_cache: true
   ```

2. **Subsequent Runs**: Use cached responses
   ```yaml
   prefill_from_export: "baseline_cache.cache"
   reuse_cached_responses: true
   ```

3. **Testing Changes**: Enable test mode
   ```yaml
   prefill_from_export: "baseline_cache.cache"
   reuse_cached_responses: true
   test_mode: true
   ```

### Performance Tips

- Use `max_saved_requests` and `max_saved_responses` to limit cache size for large evaluations
- Export cache only when needed (post-processing can be slow for large caches)
- Use local `cache_dir` for ongoing work, export cache for sharing

### Security Considerations

- Cache files contain API request/response data - handle with appropriate security
- Binary pickle format: only load cache files from trusted sources
- Consider encrypting cache exports if they contain sensitive data
