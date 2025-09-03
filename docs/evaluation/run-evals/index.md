(eval-run)=

# Run Evaluations

Step-by-step guides for different evaluation scenarios and methodologies in NeMo Eval.

## Overview

This section provides practical guides for running different types of evaluations, each optimized for specific use cases and model capabilities. Choose the evaluation type that best matches your assessment needs.

## Evaluation Flows

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`pencil;1.5em;sd-mr-1` Text Generation
:link: text-gen
:link-type: ref
Evaluate models through natural language generation for academic benchmarks, reasoning tasks, and general knowledge assessment.

:::

:::{grid-item-card} {octicon}`graph;1.5em;sd-mr-1` Log-Probability
:link: log-probability/index
:link-type: ref
Assess model confidence and uncertainty using log-probabilities for multiple-choice scenarios without text generation.

:::

:::{grid-item-card} {octicon}`code;1.5em;sd-mr-1` Code Generation
:link:
:link-type: ref
Evaluate programming capabilities through code generation, completion, and algorithmic problem solving.

:::

:::{grid-item-card} {octicon}`shield;1.5em;sd-mr-1` Safety & Security
:link:
:link-type: ref
Test AI safety, alignment, and security vulnerabilities using specialized safety harnesses and probing techniques.

:::

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Function Calling
:link:
:link-type: ref
Assess tool use capabilities, API calling accuracy, and structured output generation for agent-like behaviors.

:::

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Specialized Domains
:link:
:link-type: ref

Domain-specific evaluations for specialized use cases including multilingual, multimodal, and enterprise scenarios.

:::

::::

## Evaluation Type Selection Guide

### By Model Type

**Base Models (Pre-trained)**:
- ✅ [Log-Probability](log-probability/index.md) - No instruction following required
- ✅ [Text Generation](text-gen.md) - With academic prompting
- ❌ Avoid chat-specific evaluations

**Instruction-Tuned Models**:
- ✅ [Text Generation](text-gen.md) - Instruction following tasks
- ✅ Code Generation - Programming tasks (documentation coming soon)
- ✅ Safety Evaluation - Alignment testing (documentation coming soon)
- ✅ Function Calling - Tool use scenarios (documentation coming soon)

**Chat Models**:
- ✅ All evaluation types with appropriate chat formatting
- ✅ Conversational benchmarks and multi-turn evaluations

### By Use Case

**Academic Research**:
- [Text Generation](text-gen.md) for MMLU, reasoning benchmarks
- [Log-Probability](log-probability/index.md) for baseline comparisons
- Specialized domains for research-specific metrics (documentation coming soon)

**Production Deployment**:
- Safety evaluation for alignment validation (documentation coming soon)
- Function calling for agent capabilities (documentation coming soon)
- Code generation for programming assistants (documentation coming soon)

**Model Development**:
- [Text Generation](text-gen.md) for general capability assessment
- Multiple evaluation types for comprehensive analysis
- Custom benchmarks for specific improvements

## Configuration & Setup

### Quick Start Checklist

1. **Deploy Model**: Use [PyTriton](../../deployment/pytriton.md) or [Ray Serve](../../deployment/ray-serve.md)
2. **Install Harness**: Install required evaluation packages for your scenario
3. **Configure Parameters**: See [Configuration Parameters](../parameters.md) for optimization
4. **Run Evaluation**: Follow scenario-specific guides for detailed instructions

### Environment Requirements

```bash
# Core evaluation framework (pre-installed in NeMo container)
pip install nvidia-lm-eval>=25.6

# Optional harnesses (install as needed)
pip install nvidia-simple-evals>=25.6      # Code generation
pip install nvidia-bigcode-eval>=25.6      # Advanced code evaluation  
pip install nvidia-safety-harness>=25.6    # Safety evaluation
pip install nvidia-bfcl>=25.6             # Function calling
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

## Performance Considerations

### Evaluation Speed Optimization

- **Parallel Requests**: Configure `parallelism` based on server capacity
- **Sample Limiting**: Use `limit_samples` for quick validation
- **Endpoint Selection**: Match endpoint type to evaluation requirements
- **Resource Planning**: Allocate sufficient compute for concurrent evaluations

### Recommended Configurations

**Development/Testing**:
```python
ConfigParams(
    limit_samples=10,    # Quick validation
    parallelism=1,       # Conservative resources
    temperature=0        # Deterministic results
)
```

**Production Evaluation**:
```python
ConfigParams(
    limit_samples=None,  # Full dataset
    parallelism=8,       # High throughput
    max_retries=3        # Robust execution
)
```

## Next Steps

- **First Time**: Start with [Text Generation](text-gen.md) for general model assessment
- **Configuration**: Review [Parameters Guide](../parameters.md) for optimization options
- **Troubleshooting**: See [Evaluation Troubleshooting](../troubleshooting.md) for common issues
- **Custom Tasks**: Learn [Custom Task Configuration](../custom-tasks.md) for specialized evaluations


:::{toctree}
:hidden:

Log Probability <log-probability/index>
Text Generation <text-gen>
Log Probs <log-probs>
:::
