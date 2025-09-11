(interceptor-reasoning)=

# Reasoning

The reasoning interceptor extracts and processes chain-of-thought reasoning from model responses, enabling analysis of model reasoning patterns and supporting models that use thinking tokens.

## Overview

The `ReasoningInterceptor` handles models that generate explicit reasoning steps, typically enclosed in special tokens. It can extract reasoning content, separate it from final answers, and provide structured output for analysis.

## Configuration

### AdapterConfig Parameters

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Enable reasoning processing
    use_reasoning=True,
    
    # Reasoning token configuration
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    
    # Extraction options
    extract_reasoning=True,
    reasoning_field="reasoning"
)
```

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_reasoning=True,target.api_endpoint.adapter_config.end_reasoning_token="</think>",target.api_endpoint.adapter_config.start_reasoning_token="<think>"'
```

### YAML Configuration
```yaml
target:
  api_endpoint:
    adapter_config:
      use_reasoning: true
      start_reasoning_token: "<think>"
      end_reasoning_token: "</think>"
      extract_reasoning: true
      reasoning_field: "reasoning"
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `use_reasoning` | Enable reasoning processing | `False` | bool |
| `start_reasoning_token` | Token that marks the start of reasoning | `"<think>"` | str |
| `end_reasoning_token` | Token that marks the end of reasoning | `"</think>"` | str |
| `extract_reasoning` | Extract reasoning into separate field | `True` | bool |
| `reasoning_field` | Field name for extracted reasoning | `"reasoning"` | str |

## Processing Examples

### Input Processing
```python
# Original response from model
response_content = "<think>Let me solve this step by step. 2+2 is basic addition. 2 plus 2 equals 4.</think>The answer is 4."

# After reasoning interceptor processing
{
    "reasoning": "Let me solve this step by step. 2+2 is basic addition. 2 plus 2 equals 4.",
    "final_answer": "The answer is 4.",
    "original_content": "<think>Let me solve this step by step. 2+2 is basic addition. 2 plus 2 equals 4.</think>The answer is 4."
}
```

### Complex Reasoning Example
```python
# Multi-step reasoning
response_content = """<think>
This is a word problem. Let me break it down:
1. John has 5 apples
2. He gives away 2 apples  
3. So he has 5 - 2 = 3 apples left
</think>John has 3 apples remaining."""

# Processed output
{
    "reasoning": "This is a word problem. Let me break it down:\n1. John has 5 apples\n2. He gives away 2 apples\n3. So he has 5 - 2 = 3 apples left",
    "final_answer": "John has 3 apples remaining."
}
```

## Reasoning Analysis

### Metrics Extraction
The interceptor can extract reasoning metrics:

```json
{
    "reasoning_stats": {
        "reasoning_length": 156,
        "reasoning_tokens": 32,
        "final_answer_length": 24,
        "final_answer_tokens": 5,
        "reasoning_ratio": 0.86
    }
}
```

### Pattern Detection
```python
# Common reasoning patterns
patterns = {
    "step_by_step": "step by step|let me think|breaking it down",
    "calculation": "\\d+\\s*[+\\-*/]\\s*\\d+\\s*=",
    "conclusion": "therefore|thus|so|in conclusion"
}
```

## Use Cases

- **Chain-of-Thought Evaluation**: Evaluate models that use explicit reasoning
- **Reasoning Quality Analysis**: Assess the quality and coherence of reasoning steps
- **Performance Optimization**: Measure reasoning overhead vs. accuracy improvements
- **Educational Applications**: Understand how models approach problem-solving
- **Debugging**: Identify where models make reasoning errors

## Integration with Benchmarks

### Math Benchmarks (GSM8K, MATH)
```python
adapter_config = AdapterConfig(
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True
)
```

### Custom Reasoning Formats
```python
# For models using different reasoning tokens
adapter_config = AdapterConfig(
    use_reasoning=True,
    start_reasoning_token="[REASONING]",
    end_reasoning_token="[/REASONING]",
    extract_reasoning=True
)
```

## Best Practices

### Token Configuration
- Use consistent reasoning tokens across evaluations
- Choose tokens that are unlikely to appear in regular content
- Consider model-specific token preferences

### Analysis Workflow
```python
# Complete reasoning analysis setup
adapter_config = AdapterConfig(
    use_reasoning=True,
    start_reasoning_token="<think>",
    end_reasoning_token="</think>",
    extract_reasoning=True,
    use_request_logging=True,  # Log for analysis
    use_response_logging=True,
    save_responses=True        # Save for offline analysis
)
```
