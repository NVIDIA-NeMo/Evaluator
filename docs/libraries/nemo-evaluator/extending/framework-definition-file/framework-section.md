(framework-section)=

# Framework Section

The `framework` section contains basic identification and metadata for your evaluation framework.

## Structure

```yaml
framework:
  name: example-evaluation-framework         # Internal framework identifier
  pkg_name: example_evaluation_framework     # Python package name
  full_name: Example Evaluation Framework    # Human-readable display name
  description: A comprehensive example...    # Detailed description
  url: https://github.com/example/...        # Original repository URL
```

## Fields

### name

**Type**: String  
**Required**: Yes

Unique identifier used internally by the system. This should be a lowercase, hyphenated string that identifies your framework.

**Example**:
```yaml
name: bigcode-evaluation-harness
```

### pkg_name

**Type**: String  
**Required**: Yes

Python package name for your framework. This typically matches the `name` field but uses underscores instead of hyphens to follow Python naming conventions.

**Example**:
```yaml
pkg_name: bigcode_evaluation_harness
```

### full_name

**Type**: String  
**Required**: Recommended

Human-readable name displayed in the UI and documentation. Use proper capitalization and spacing.

**Example**:
```yaml
full_name: BigCode Evaluation Harness
```

### description

**Type**: String  
**Required**: Recommended

Comprehensive description of the framework's purpose, capabilities, and use cases. This helps users understand when to use your framework.

**Example**:
```yaml
description: A comprehensive evaluation harness for code generation models, supporting multiple programming languages and diverse coding tasks.
```

### url

**Type**: String (URL)  
**Required**: Recommended

Link to the original benchmark or framework repository. This provides users with access to more documentation and source code.

**Example**:
```yaml
url: https://github.com/bigcode-project/bigcode-evaluation-harness
```

## Best Practices

- Use consistent naming across `name`, `pkg_name`, and `full_name`
- Keep the `name` field URL-friendly (lowercase, hyphens)
- Write clear, concise descriptions that highlight unique features
- Link to the canonical upstream repository when available
- Verify that the URL is accessible and up-to-date

## Minimal Requirements

At minimum, an FDF requires the `name` and `pkg_name` fields. However, including `full_name`, `description`, and `url` is strongly recommended for better documentation and user experience.

