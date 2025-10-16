(interceptor-response-stats)=
# Response Stats Interceptor

## Overview

The `ResponseStatsInterceptor` collects comprehensive aggregated statistics from API responses for metrics collection and analysis. It tracks detailed metrics about token usage, response patterns, performance characteristics, and API behavior throughout the evaluation process.

This interceptor is essential for understanding API performance, cost analysis, and monitoring evaluation runs. It provides both real-time aggregated statistics and detailed per-request tracking capabilities.

**Key Statistics Tracked:**

- Token usage (prompt, completion, total) with averages and maximums
- Response status codes and counts
- Finish reasons and stop reasons
- Tool calls and function calls counts
- Response latency (average and maximum)
- Total response count and successful responses
- Inference run times and timing analysis

## Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.tracking_requests_stats=True,target.api_endpoint.adapter_config.response_stats_cache=/tmp/response_stats_interceptor,target.api_endpoint.adapter_config.logging_aggregated_stats_interval=100'
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "response_stats"
          enabled: true
          config:
            # Default configuration - collect all statistics
            collect_token_stats: true
            collect_finish_reasons: true
            collect_tool_calls: true
            save_individuals: true
            cache_dir: "/tmp/response_stats_interceptor"
            logging_aggregated_stats_interval: 100
        - name: "endpoint"
          enabled: true
          config: {}
```

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "response_stats"
          enabled: true
          config:
            # Minimal configuration - only basic stats
            collect_token_stats: false
            collect_finish_reasons: false
            collect_tool_calls: false
            save_individuals: false
            logging_aggregated_stats_interval: 50
        - name: "endpoint"
          enabled: true
          config: {}
```

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: "endpoint"
          enabled: true
          config: {}
        - name: "response_stats"
          enabled: true
          config:
            # Custom configuration with periodic saving
            collect_token_stats: true
            collect_finish_reasons: true
            collect_tool_calls: true
            stats_file_saving_interval: 100
            save_individuals: true
            cache_dir: "/custom/stats/cache"
            logging_aggregated_stats_interval: 25

```

## Configuration Options

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collect_token_stats` | `bool` | `true` | Whether to collect token statistics (prompt, completion, total tokens) |
| `collect_finish_reasons` | `bool` | `true` | Whether to collect and track finish reasons from API responses |
| `collect_tool_calls` | `bool` | `true` | Whether to collect tool call and function call statistics |
| `stats_file_saving_interval` | `int` | `None` | How often (every N responses) to save stats to file. If None, only saves via post_eval_hook |
| `save_individuals` | `bool` | `true` | Whether to save individual request statistics. If false, only saves aggregated stats |
| `cache_dir` | `str` | `"/tmp/response_stats_interceptor"` | Custom cache directory for storing response statistics |
| `logging_aggregated_stats_interval` | `int` | `100` | How often (every N responses) to log aggregated statistics to console |

## Behavior

### Statistics Collection
The interceptor automatically collects statistics from successful API responses (HTTP 200) and tracks basic information for all responses regardless of status code.

**For Successful Responses (200):**
- Parses JSON response body
- Extracts token usage from `usage` field
- Collects finish reasons from `choices[].finish_reason`
- Counts tool calls and function calls
- Calculates running averages and maximums

**For All Responses:**
- Tracks status code distribution
- Measures response latency
- Records response timestamps
- Maintains response counts

### Data Storage
- **Aggregated Stats**: Continuously updated running statistics stored in cache
- **Individual Stats**: Per-request details stored with request IDs (if enabled)
- **Metrics File**: Final statistics saved to `eval_factory_metrics.json`
- **Thread Safety**: All operations are thread-safe using locks

### Timing Analysis
- Tracks inference run times across multiple evaluation runs
- Calculates time from first to last request per run
- Estimates time to first request from adapter initialization
- Provides detailed timing breakdowns for performance analysis

## Statistics Output

### Aggregated Statistics
```json
{
  "response_stats": {
    "description": "Response statistics saved during processing",
    "avg_prompt_tokens": 150.5,
    "avg_total_tokens": 200.3,
    "avg_completion_tokens": 49.8,
    "avg_latency_ms": 1250.2,
    "max_prompt_tokens": 300,
    "max_total_tokens": 450,
    "max_completion_tokens": 150,
    "max_latency_ms": 3000,
    "count": 1000,
    "successful_count": 995,
    "tool_calls_count": 50,
    "function_calls_count": 25,
    "finish_reason": {
      "stop": 800,
      "length": 150,
      "tool_calls": 45
    },
    "status_codes": {
      "200": 995,
      "429": 3,
      "500": 2
    },
    "inference_time": 45.6,
    "run_id": 0
  }
}
```

### Individual Request Statistics (if enabled)
```json
{
  "request_id": "req_123",
  "timestamp": 1698765432.123,
  "status_code": 200,
  "prompt_tokens": 150,
  "total_tokens": 200,
  "completion_tokens": 50,
  "finish_reason": "stop",
  "tool_calls_count": 0,
  "function_calls_count": 0,
  "run_id": 0
}
```


## Common Use Cases

- **Cost Analysis**: Track token usage patterns to estimate API costs
- **Performance Monitoring**: Monitor response times and throughput
- **Quality Assessment**: Analyze finish reasons and response patterns
- **Tool Usage Analysis**: Track function and tool call frequencies
- **Debugging**: Individual request tracking for troubleshooting
- **Capacity Planning**: Understand API usage patterns and limits
- **A/B Testing**: Compare statistics across different configurations
- **Production Monitoring**: Real-time visibility into API behavior

## Integration Notes

- **Post-Evaluation Hook**: Automatically saves final statistics after evaluation completes
- **Cache Persistence**: Statistics survive across runs and can be aggregated
- **Thread Safety**: Safe for concurrent request processing
- **Memory Efficient**: Uses running averages to avoid storing all individual values
- **Caching Strategy**: Handles cache hits by skipping statistics collection to avoid double-counting
