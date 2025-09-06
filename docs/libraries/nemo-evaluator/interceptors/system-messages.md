# System Messages

The system message interceptor injects custom system prompts into evaluation requests, enabling consistent prompting and role-specific behavior across evaluations.

## Overview

The `SystemMessageInterceptor` modifies incoming requests to include custom system messages. This is particularly useful for standardizing model behavior, implementing specific personas, or adding evaluation-specific instructions.

## Configuration

### AdapterConfig Parameters

```python
from nemo_evaluator.adapters.adapter_config import AdapterConfig

adapter_config = AdapterConfig(
    # Enable system message injection
    use_system_prompt=True,
    
    # Custom system message
    custom_system_prompt="You are a helpful AI assistant. Think step by step before answering."
)
```

### CLI Configuration
```bash
--overrides 'target.api_endpoint.adapter_config.use_system_prompt=True,target.api_endpoint.adapter_config.custom_system_prompt="You are a helpful assistant."'
```

### YAML Configuration
```yaml
target:
  api_endpoint:
    adapter_config:
      use_system_prompt: true
      custom_system_prompt: "You are a helpful AI assistant. Think step by step before answering."
```

## Configuration Options

| Parameter | Description | Default | Type |
|-----------|-------------|---------|------|
| `use_system_prompt` | Enable system message injection | `False` | bool |
| `custom_system_prompt` | Custom system message content | `None` | str |

## Message Processing

### Chat API Format
```python
# Original request
{
    "messages": [
        {"role": "user", "content": "What is 2+2?"}
    ]
}

# After system message interceptor
{
    "messages": [
        {"role": "system", "content": "You are a helpful AI assistant. Think step by step before answering."},
        {"role": "user", "content": "What is 2+2?"}
    ]
}
```

### Completions API Format
```python
# Original request
{
    "prompt": "What is 2+2?"
}

# After system message interceptor
{
    "prompt": "You are a helpful AI assistant. Think step by step before answering.\n\nWhat is 2+2?"
}
```

## Use Cases

### Consistent Evaluation Behavior
```python
# Ensure all models use the same system prompt
adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt="You are an expert problem solver. Provide clear, accurate answers."
)
```

### Role-Specific Evaluation
```python
# Math tutor persona for mathematical benchmarks
adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt="You are a mathematics tutor. Explain your reasoning step by step and show all calculations clearly."
)
```

### Benchmark-Specific Instructions
```python
# Code generation specific instructions
adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt="You are a software engineer. Write clean, well-documented code with appropriate comments."
)
```

### Safety and Alignment
```python
# Safety-focused system message
adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt="You are a helpful, harmless, and honest AI assistant. Always prioritize safety and accuracy in your responses."
)
```

## Advanced Usage

### Dynamic System Messages
```python
# Context-aware system message selection
def get_system_prompt(benchmark_type):
    prompts = {
        "math": "You are a mathematics expert. Show your work step by step.",
        "coding": "You are a software engineer. Write clean, efficient code.",
        "reasoning": "You are a logical reasoning expert. Think through problems systematically.",
        "safety": "You are a helpful assistant. Always prioritize safety and ethical considerations."
    }
    return prompts.get(benchmark_type, "You are a helpful AI assistant.")

adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt=get_system_prompt("math")
)
```

### Multi-Language Support
```python
# Language-specific system messages
system_prompts = {
    "en": "You are a helpful AI assistant.",
    "es": "Eres un asistente de IA útil.",
    "fr": "Vous êtes un assistant IA utile.",
    "de": "Sie sind ein hilfreicher KI-Assistent."
}

adapter_config = AdapterConfig(
    use_system_prompt=True,
    custom_system_prompt=system_prompts["en"]
)
```

## Best Practices

### Clear and Specific Instructions
```python
# Good: Specific and actionable
custom_system_prompt = """You are an expert data analyst. When answering questions:
1. Analyze the data carefully
2. Show your calculations
3. Explain your reasoning
4. Provide confidence levels for your conclusions"""

# Avoid: Vague or generic
custom_system_prompt = "Be helpful and answer questions well."
```

### Consistency Across Evaluations
```python
# Use consistent system messages for comparable results
EVALUATION_SYSTEM_PROMPTS = {
    "academic": "You are an academic researcher. Provide thorough, well-reasoned responses.",
    "practical": "You are a practical problem solver. Focus on actionable solutions.",
    "creative": "You are a creative thinker. Generate innovative and original ideas."
}
```

### Evaluation-Specific Customization
```python
# Customize for specific evaluation needs
adapter_configs = {
    "gsm8k": AdapterConfig(
        use_system_prompt=True,
        custom_system_prompt="You are a math tutor. Solve problems step by step."
    ),
    "humaneval": AdapterConfig(
        use_system_prompt=True, 
        custom_system_prompt="You are a software engineer. Write clean Python code."
    ),
    "mmlu": AdapterConfig(
        use_system_prompt=True,
        custom_system_prompt="You are a knowledgeable expert. Choose the best answer."
    )
}
```

The system message interceptor is essential for standardizing model behavior and ensuring consistent evaluation conditions across different benchmarks and model endpoints.
