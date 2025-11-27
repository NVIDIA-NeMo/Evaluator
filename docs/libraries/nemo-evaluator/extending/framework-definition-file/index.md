(framework-definition-file)=

# Framework Definition File (FDF)

Framework Definition Files are YAML configuration files that integrate evaluation frameworks into NeMo Evaluator. They define framework metadata, execution commands, and evaluation tasks.

**New to FDFs?** Learn about {ref}`the concepts and architecture <fdf-concept>` before creating one.

## Prerequisites

Before creating an FDF, you should:

- Understand YAML syntax and structure
- Be familiar with your evaluation framework's CLI interface
- Have basic knowledge of Jinja2 templating
- Know the API endpoint types your framework supports

## Getting Started

**Creating your first FDF?** Follow this sequence:

<!-- 1. Start with the {ref}`create-framework-definition-file` tutorial for a hands-on walkthrough -->
1. {ref}`framework-section` - Define framework metadata
2. {ref}`defaults-section` - Configure command templates and parameters
3. {ref}`evaluations-section` - Define evaluation tasks
4. {ref}`integration` - Integrate with Eval Factory

**Need help?** Refer to {ref}`fdf-troubleshooting` for debugging common issues.

## Complete Example

The FDF follows a hierarchical structure with three main sections. Here's a minimal but complete example:

```yaml
# 1. Framework Identification
framework:
  name: my-custom-eval
  pkg_name: my_custom_eval
  full_name: My Custom Evaluation Framework
  description: Evaluates domain-specific capabilities
  url: https://github.com/example/my-eval

# 2. Default Command and Configuration
defaults:
  command: >-
    {% if target.api_endpoint.api_key is not none %}export API_KEY=${{target.api_endpoint.api_key}} && {% endif %}
    my-eval-cli --model {{target.api_endpoint.model_id}} 
                --task {{config.params.task}}
                --output {{config.output_dir}}
  
  config:
    params:
      temperature: 0.0
      max_new_tokens: 1024
  
  target:
    api_endpoint:
      type: chat
      supported_endpoint_types:
        - chat
        - completions

# 3. Evaluation Types
evaluations:
  - name: my_task_1
    description: First evaluation task
    defaults:
      config:
        type: my_task_1
        supported_endpoint_types:
          - chat
        params:
          task: my_task_1
```

## Reference Documentation

::::{grid} 1 2 2 2
:gutter: 1 1 1 2

:::{grid-item-card} {octicon}`package;1.5em;sd-mr-1` Framework Section
:link: framework-section
:link-type: ref
Define framework metadata including name, package information, and repository URL.
:::

:::{grid-item-card} {octicon}`list-unordered;1.5em;sd-mr-1` Defaults Section
:link: defaults-section
:link-type: ref
Configure default parameters, command templates, and target endpoint settings.
:::

:::{grid-item-card} {octicon}`checklist;1.5em;sd-mr-1` Evaluations Section
:link: evaluations-section
:link-type: ref
Define specific evaluation types with task-specific configurations and parameters.
:::

:::{grid-item-card} {octicon}`telescope;1.5em;sd-mr-1` Advanced Features
:link: advanced-features
:link-type: ref
Use conditionals, parameter inheritance, and dynamic configuration in your FDF.
:::

:::{grid-item-card} {octicon}`plug;1.5em;sd-mr-1` Integration
:link: integration
:link-type: ref
Learn how to integrate your FDF with the Eval Factory system.
:::

:::{grid-item-card} {octicon}`question;1.5em;sd-mr-1` Troubleshooting
:link: fdf-troubleshooting
:link-type: ref
Debug common issues with template errors, parameters, and validation.
:::

::::

## Related Documentation

- {ref}`eval-custom-tasks` - Learn how to create custom evaluation tasks
- {ref}`extending-evaluator` - Overview of extending the NeMo Evaluator
- {ref}`parameter-overrides` - Using parameter overrides in evaluations

:::{toctree}
:maxdepth: 1
:hidden:

Framework Section <framework-section>
Defaults Section <defaults-section>
Evaluations Section <evaluations-section>
Advanced Features <advanced-features>
Integration <integration>
Troubleshooting <fdf-troubleshooting>
:::

