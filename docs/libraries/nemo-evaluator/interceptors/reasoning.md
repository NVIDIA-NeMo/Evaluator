(interceptor-reasoning)=

# Reasoning

## Overview

The `ResponseReasoningInterceptor` handles models that generate explicit reasoning steps, typically enclosed in special tokens. It removes reasoning content from the final response and tracks reasoning metrics for analysis.

## Configuration

### Python Configuration

### CLI Configuration

```bash
--overrides 'target.api_endpoint.adapter_config.use_reasoning=True,target.api_endpoint.adapter_config.end_reasoning_token="</think>",target.api_endpoint.adapter_config.start_reasoning_token="<think>"'
```

### YAML Configuration

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: reasoning
          config:
            start_reasoning_token: "<think>"
            end_reasoning_token: "</think>"
            add_reasoning: true
            enable_reasoning_tracking: true
        - name: "endpoint"
          enabled: true
          config: {}
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `start_reasoning_token` | Token that marks the start of reasoning section | `"<think>"` | str \| None |
| `end_reasoning_token` | Token that marks the end of reasoning section | `"</think>"` | str |
| `add_reasoning` | Whether to add reasoning information | `True` | bool |
| `migrate_reasoning_content` | Migrate reasoning_content to content field with tokens | `False` | bool |
| `enable_reasoning_tracking` | Enable reasoning tracking and logging | `True` | bool |
| `include_if_not_finished` | Include reasoning content if reasoning is not finished (end token not found) | `True` | bool |
| `stats_file_saving_interval` | How often (every N responses) to save stats to file | `None` | int \| None |
| `enable_caching` | Whether to enable caching of reasoning statistics | `True` | bool |
| `cache_dir` | Custom cache directory for reasoning stats | `"/tmp/reasoning_interceptor"` | str |
| `logging_aggregated_stats_interval` | How often (every N responses) to log aggregated reasoning statistics | `100` | int |

## Processing Examples

### Basic Reasoning Stripping

```python
# Original response from model
original_content = "<think>Let me solve this step by step. 2+2 is basic addition. 2 plus 2 equals 4.</think>The answer is 4."

# After reasoning interceptor processing
# The content field has reasoning removed
processed_content = "The answer is 4."
```

### Multi-Step Reasoning

```python
# Original response with multi-line reasoning
original_content = """<think>
This is a word problem. Let me break it down:
1. John has 5 apples
2. He gives away 2 apples  
3. So he has 5 - 2 = 3 apples left
</think>John has 3 apples remaining."""

# After processing: reasoning tokens and content are removed
processed_content = "John has 3 apples remaining."
```

## Tracked Metrics

The interceptor automatically tracks the following statistics:

| Metric | Description |
|--------|-------------|
| `total_responses` | Total number of responses processed |
| `responses_with_reasoning` | Number of responses containing reasoning content |
| `reasoning_finished_count` | Number of responses where reasoning completed (end token found) |
| `reasoning_started_count` | Number of responses where reasoning started |
| `avg_reasoning_words` | Average word count in reasoning content |
| `avg_reasoning_tokens` | Average token count in reasoning content |
| `avg_original_content_words` | Average word count in original content (before processing) |
| `avg_updated_content_words` | Average word count in updated content (after processing) |
| `avg_updated_content_tokens` | Average token count in updated content |
| `max_reasoning_words` | Maximum word count in reasoning content |
| `max_reasoning_tokens` | Maximum token count in reasoning content |
| `max_original_content_words` | |
| `max_updated_content_words` | |
| `max_updated_content_tokens` | Maximum token count in updated content |
| `total_reasoning_words` | Total word count across all reasoning content |
| `total_reasoning_tokens` | Total token count across all reasoning content |
| `total_original_content_words` | Total word count in original content (before processing) |
| `total_updated_content_words` | Total word count in updated content (after processing) |
| `total_updated_content_tokens` | Total token count in updated content |

These statistics are saved to `eval_factory_metrics.json` under the `reasoning` key after evaluation completes.

## Example: Custom Reasoning Tokens

```yaml
target:
  api_endpoint:
    adapter_config:
      interceptors:
        - name: reasoning
          config:
            start_reasoning_token: "[REASONING]"
            end_reasoning_token: "[/REASONING]"
            add_reasoning: true
            enable_reasoning_tracking: true
        - name: "endpoint"
          enabled: true
          config: {}
```
