(eval-run)=

# Run Evaluations

Step-by-step guides for different evaluation scenarios and methodologies in NeMo Eval.

## Overview

This section provides practical guides for running different types of evaluations, each optimized for specific use cases and model capabilities. Choose the evaluation type that best matches your assessment needs.

## Before You Start

1. Ensure you have completed the initial getting started guides for [installation]() and [quickstart]().
1. Select an existing OpenAI-compatible endpoint, or deploy one with [PyTriton](../../deployment/pytriton.md) or [Ray Serve](../../deployment/ray-serve.md)
3. Review the [evaluation parameters](eval-parameters) available for optimization.

::::{tab-set}

:::{tab-item}

### Environment Requirements

```bash
# Core evaluation framework (pre-installed in NeMo container)
pip install nvidia-lm-eval==25.7.1

# Optional harnesses (install as needed)
pip install nvidia-simple-evals>=25.6      # Baseline/simple evaluations
pip install nvidia-bigcode-eval>=25.6      # Advanced code evaluation  
pip install nvidia-safety-harness>=25.6    # Safety evaluation
pip install nvidia-bfcl>=25.6              # Function calling
pip install nvidia-eval-factory-garak>=25.6  # Security scanning
```

### Authentication Setup

Some evaluations require additional authentication:

```bash
# Hugging Face token for gated datasets
export HF_TOKEN="your_hf_token"

# NVIDIA Build API key for judge models (safety evaluation)
export JUDGE_API_KEY="your_nvidia_api_key"
```

---

## Evaluations

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Text Generation
:link: text-gen
:link-type: ref
Evaluate models through natural language generation for academic benchmarks, reasoning tasks, and general knowledge assessment.

:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Log-Probability
:link: log-probability
:link-type: ref
Assess model confidence and uncertainty using log-probabilities for multiple-choice scenarios without text generation.

:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link: code-generation
:link-type: ref
Evaluate programming capabilities through code generation, completion, and algorithmic problem solving.

:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Safety & Security
:link: safety-security
:link-type: ref
Test AI safety, alignment, and security vulnerabilities using specialized safety harnesses and probing techniques.

:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Function Calling
:link: function-calling
:link-type: ref
Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors.

:::

::::

### Selection Guide

::::{tab-set}
:::{tab-item} By Model Type

```{list-table}
:header-rows: 1
:widths: 25 75

* - Model Type
  - Recommended Evaluations
* - Base Models (Pre-trained)
  -
    - ✅ [Log-Probability](log-probability/index.md) - No instruction following required
    - ✅ [Text Generation](text-gen.md) - With academic prompting
    - ❌ Avoid chat-specific evaluations
* - Instruction-Tuned Models
  -
    - ✅ [Text Generation](text-gen.md) - Instruction following tasks
    - ✅ [Code Generation](code-generation.md) - Programming tasks and algorithmic problem solving
    - ✅ [Safety & Security](safety-security.md) - Alignment testing and vulnerability scanning
    - ✅ [Function Calling](function-calling.md) - Tool use scenarios and API integration
* - Chat Models
  -
    - ✅ All evaluation types with appropriate chat formatting
    - ✅ Conversational benchmarks and multi-turn evaluations
```

:::

:::{tab-item} By Use Case

```{list-table}
:header-rows: 1
:widths: 25 75

* - Use Case
  - Recommended Evaluations
* - Academic Research
  -
    - [Text Generation](text-gen.md) for MMLU, reasoning benchmarks
    - [Log-Probability](log-probability/index.md) for baseline comparisons
    - Specialized domains for research-specific metrics (documentation coming soon)
* - Production Deployment
  -
    - [Safety & Security](safety-security.md) for alignment validation and vulnerability testing
    - [Function Calling](function-calling.md) for agent capabilities and tool use
    - [Code Generation](code-generation.md) for programming assistants and code completion
* - Model Development
  -
    - [Text Generation](text-gen.md) for general capability assessment
    - Multiple evaluation types for comprehensive analysis
    - Custom benchmarks for specific improvements
```

:::

::::

## Performance Considerations

### Evaluation Speed Optimization

- **Parallel Requests**: Configure `parallelism` based on server capacity
- **Sample Limiting**: Use `limit_samples` for quick validation
- **Endpoint Selection**: Match endpoint type to evaluation requirements
- **Resource Planning**: Allocate sufficient compute for concurrent evaluations

### Recommended Configurations

::::{tab-set}

:::{tab-item} Development & Testing
```python
ConfigParams(
    limit_samples=10,    # Quick validation
    parallelism=1,       # Conservative resources
    temperature=0        # Deterministic results
)
```

:::

:::{tab-item} Production

```python
ConfigParams(
    limit_samples=None,  # Full dataset
    parallelism=8,       # High throughput
    max_retries=3        # Robust execution
)
```

:::

::::

:::{toctree}
:hidden:

Log Probability <log-probability/index>
Text Generation <text-gen>
Code Generation <code-generation>
Function Calling <function-calling>
Safety & Security <safety-security>
Log Probs <logprobs>
:::
