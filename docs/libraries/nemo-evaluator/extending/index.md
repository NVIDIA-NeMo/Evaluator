(extending-evaluator)=

# Extending NeMo Evaluator

Extend NeMo Evaluator with custom benchmarks, evaluation frameworks, and integrations. Learn how to define new evaluation frameworks and integrate them into the NeMo Evaluator ecosystem using standardized configuration patterns.

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Framework Definition File
:link: framework-definition-file
:link-type: ref

Learn how to create Framework Definition Files (FDF) to integrate custom evaluation frameworks and benchmarks into the NeMo Evaluator ecosystem.
:::

:::{grid-item-card} {octicon}`beaker;1.5em;sd-mr-1` Bring Your Own Benchmark (BYOB)
:link: byob
:link-type: ref

Create custom evaluation benchmarks in ~12 lines of Python with decorators, built-in scorers, LLM-as-Judge, and one-command containerization.
:::

::::

## Extension Patterns

NeMo Evaluator supports several patterns for extending functionality:

### Framework Definition Files (FDF)

The primary extension mechanism uses YAML configuration files to define:

- Framework metadata and dependencies
- Default configurations and parameters
- Evaluation types and task definitions
- Container integration specifications

### Bring Your Own Benchmark (BYOB)

A decorator-based approach for creating custom benchmarks in Python:

- Define benchmarks with `@benchmark` and `@scorer` decorators
- Use built-in scorers or write custom scoring functions
- Evaluate subjective qualities with LLM-as-Judge
- Containerize and deploy with a single CLI command

### Integration Benefits

- **Standardization**: Follow established patterns for configuration and execution
- **Reproducibility**: Leverage the same deterministic configuration system
- **Compatibility**: Work seamlessly with existing launchers and exporters
- **Community**: Share frameworks through the standard FDF format

## Start with Extensions

**Want a quick custom benchmark?** Start with {ref}`BYOB <byob>` to create a benchmark in ~12 lines of Python.

**Building a production framework?** Follow these steps:

1. **Review Existing Frameworks**: Study existing FDF files to understand the structure
2. **Define Your Framework**: Create an FDF that describes your evaluation framework
3. **Test Integration**: Validate that your framework works with NeMo Evaluator workflows
4. **Container Packaging**: Package your framework as a container for distribution

For detailed reference documentation, refer to {ref}`framework-definition-file`.

:::{toctree}
:caption: Extending NeMo Evaluator
:hidden:

Framework Definition File <framework-definition-file/index>
Bring Your Own Benchmark <byob/index>
:::
