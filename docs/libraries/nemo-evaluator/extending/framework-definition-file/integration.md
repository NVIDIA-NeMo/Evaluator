(integration)=

# Integration with Eval Factory

This section describes how to integrate your Framework Definition File with the Eval Factory system.

## File Location

Place your FDF in the `core_evals/<framework_name>/` directory of your framework package:

```
your-framework/
 core_evals/
    your_framework/
        framework.yml           # This is your FDF
        output.py               # Output parser (custom)
        __init__.py             # Empty init file
 setup.py                        # Package configuration
 README.md                       # Framework documentation
```

### Directory Structure Explanation

**core_evals/**: Root directory for evaluation framework definitions. This directory name is required by the Eval Factory system.

**your_framework/**: Subdirectory named after your framework (must match `framework.name` from your FDF).

**framework.yml**: Your Framework Definition File. This exact filename is required.

**output.py**: Custom output parser for processing evaluation results. This file should implement the parsing logic specific to your framework's output format.

**__init__.py**: Empty initialization file to make the directory a Python package.

## Validation

The FDF is validated by the NeMo Evaluator system when loaded. Validation occurs through Pydantic models that ensure:

- Required fields are present (`name`, `pkg_name`, `command`)
- Parameter types are correct (strings, integers, floats, lists)
- Template syntax is valid (Jinja2 parsing)
- Configuration consistency (endpoint types, parameter references)

### Validation Checks

**Schema Validation**: Pydantic models ensure required fields exist and have correct types when the FDF is parsed.

**Template Validation**: Jinja2 templates are rendered with `StrictUndefined`, which raises errors for undefined variables.

**Reference Validation**: Template variables must reference valid fields in the `Evaluation` model (`config`, `target`, `framework_name`, `pkg_name`).

**Consistency Validation**: Endpoint types and parameters should be consistent across framework defaults and evaluation-specific configurations.

## Registration

Once your FDF is properly located and validated, the Eval Factory system automatically:

1. Discovers your framework during initialization
2. Parses the FDF and validates its structure
3. Registers available evaluation types
4. Makes your framework available via CLI commands

## Using Your Framework

After successful integration, you can use your framework with the Eval Factory CLI:

```bash
# List available frameworks and tasks
nemo-evaluator ls

# Run an evaluation
nemo-evaluator run_eval --eval_type your_evaluation --model_id my-model ...
```

## Package Configuration

Ensure your `setup.py` or `pyproject.toml` includes the FDF in package data:

```python
from setuptools import setup, find_packages

setup(
    name="your-framework",
    packages=find_packages(),
    package_data={
        "core_evals": ["**/*.yml"],
    },
    include_package_data=True,
)
```

```toml
[tool.setuptools.package-data]
core_evals = ["**/*.yml"]
```

## Best Practices

- Follow the exact directory structure and naming conventions
- Test your FDF validation locally before deployment
- Document your framework's output format in README.md
- Include example configurations in your documentation
- Provide sample commands for common use cases
- Version your FDF changes alongside framework updates
- Keep the FDF synchronized with your framework's capabilities

