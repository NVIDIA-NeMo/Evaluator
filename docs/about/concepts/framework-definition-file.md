(fdf-concept)=

# Framework Definition Files

::::{note}
**Who needs this?** This documentation is for framework developers and organizations creating custom evaluation frameworks. If you're running existing evaluation tasks using {ref}`nemo-evaluator-launcher <lib-launcher>` (NeMo Evaluator Launcher CLI) or {ref}`nemo-evaluator <nemo-evaluator-cli>` (NeMo Evaluator CLI), you don't need to create FDFs—they're already provided by framework packages.
::::

A Framework Definition File (FDF) is a YAML configuration file that serves as the single source of truth for integrating evaluation frameworks into the NeMo Evaluator ecosystem. FDFs define how evaluation frameworks are configured, executed, and integrated with the Eval Factory system.

## What an FDF Defines

An FDF specifies five key aspects of an evaluation framework:

- **Framework metadata**: Name, description, package information, and repository URL
- **Default configurations**: Parameters, commands, and settings that apply across all evaluations within that framework
- **Evaluation types**: Available evaluation tasks and their specific configurations
- **Execution commands**: Jinja2-templated commands for running evaluations with dynamic parameter injection
- **API compatibility**: Supported endpoint types (chat, completions, vlm, embedding) and their configurations

## How FDFs Integrate with NeMo Evaluator

FDFs sit at the integration point between your evaluation framework's CLI and NeMo Evaluator's orchestration system:

```{mermaid}
graph LR
    A[User runs<br/>nemo-evaluator] --> B[System loads<br/>framework.yml]
    B --> C[Merges defaults +<br/> user evaluation config]
    C --> D[Renders Jinja2<br/>command template]
    D --> E[Executes your<br/>CLI command]
    E --> F[Parses output]
    
    style B fill:#e1f5fe
    style D fill:#fff3e0
    style E fill:#f3e5f5
```

**The workflow:**

1. When you run `nemo-evaluator` (see {ref}`nemo-evaluator-cli`), the system discovers and loads your FDF (`framework.yml`)
2. Configuration values are merged from framework defaults, evaluation-specific settings, and user overrides (see {ref}`parameter-overrides`)
3. The system renders the Jinja2 command template with the merged configuration
4. Your framework's CLI is executed with the generated command
5. Results are parsed and processed by the system

This architecture allows you to integrate any evaluation framework that exposes a CLI interface, without modifying NeMo Evaluator's core code.

## Key Concepts

### Jinja2 Templating

FDFs use Jinja2 template syntax to inject configuration values dynamically into command strings. Variables are referenced using `{{variable}}` syntax:

```yaml
command: >-
  my-eval-cli --model {{target.api_endpoint.model_id}} 
              --task {{config.params.task}}
              --output {{config.output_dir}}
```

At runtime, these variables are replaced with actual values from the configuration.

### Parameter Inheritance

Configuration values cascade through multiple layers, with later layers overriding earlier ones:

1. **Framework defaults**: Base configuration in the FDF's `defaults` section
2. **Evaluation defaults**: Task-specific overrides in the `evaluations` section
3. **User configuration**: Values from run configuration files
4. **CLI overrides**: Command-line arguments passed at runtime

This inheritance model allows you to define sensible defaults while giving users full control over specific runs. For detailed examples and patterns, see {ref}`advanced-features`.

### Endpoint Types

Evaluations declare which API endpoint types they support (see {ref}`evaluation-model` for details). NeMo Evaluator uses adapters to translate between different API formats:

- **`chat`**: OpenAI-compatible chat completions (messages with roles)
- **`completions`**: Text completion endpoints (prompt in, text out)
- **`vlm`**: Vision-language models (text + image inputs)
- **`embedding`**: Embedding generation endpoints

Your FDF specifies which types each evaluation supports, and the system validates compatibility at runtime.

### Validation

FDFs are validated when loaded to catch configuration errors early:

- **Schema validation**: Pydantic models ensure required fields exist and have correct types
- **Template validation**: Jinja2 templates are parsed with `StrictUndefined` to catch undefined variables
- **Reference validation**: Template variables must reference valid fields in the configuration model
- **Consistency validation**: Endpoint types and parameters are checked for consistency

Validation failures produce clear error messages that help you fix configuration issues before runtime. For common validation errors and solutions, see {ref}`fdf-troubleshooting`.

## File Structure

An FDF follows a three-section hierarchical structure:

```yaml
framework:          # Framework identification and metadata
  name: my-eval
  pkg_name: my_eval
  full_name: My Evaluation Framework
  description: Evaluates specific capabilities
  url: https://github.com/example/my-eval

defaults:           # Default configurations and commands
  command: >-
    my-eval-cli --model {{target.api_endpoint.model_id}}
  config:
    params:
      temperature: 0.0
  target:
    api_endpoint:
      type: chat

evaluations:        # Available evaluation types
  - name: task_1
    description: First task
    defaults:
      config:
        params:
          task: task_1
```

## Next Steps

Ready to create your own FDF? Refer to {ref}`framework-definition-file` for detailed reference documentation and practical guidance on building Framework Definition Files.
