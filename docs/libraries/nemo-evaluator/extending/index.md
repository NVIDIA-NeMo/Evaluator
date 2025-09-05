# Extending NeMo Evaluator

Extend NeMo Evaluator with custom benchmarks, evaluation frameworks, and integrations. Learn how to define new evaluation frameworks and integrate them into the NeMo Evaluator ecosystem using standardized configuration patterns.

::::{grid} 1 1 1 1
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`tools;1.5em;sd-mr-1` Framework Definition File
:link: framework_definition_file
:link-type: doc

Learn how to create Framework Definition Files (FDF) to integrate custom evaluation frameworks and benchmarks into the NeMo Evaluator ecosystem.
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

### Integration Benefits

- **Standardization**: Follow established patterns for configuration and execution
- **Reproducibility**: Leverage the same deterministic configuration system
- **Compatibility**: Work seamlessly with existing launchers and exporters
- **Community**: Share frameworks through the standard FDF format

## Getting Started with Extensions

1. **Review Existing Frameworks**: Study existing FDF files to understand the structure
2. **Define Your Framework**: Create an FDF that describes your evaluation framework
3. **Test Integration**: Validate that your framework works with NeMo Evaluator workflows
4. **Container Packaging**: Package your framework as a container for distribution

For detailed implementation guidance, see the [Framework Definition File](framework_definition_file.md) documentation.

:::{toctree}
:caption: Extending NeMo Evaluator
:hidden:

Framework Definition File <framework_definition_file>
:::
