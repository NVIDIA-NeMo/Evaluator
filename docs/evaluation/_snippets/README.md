# Documentation Snippets

This directory contains **executable** code snippets that are included in multiple documentation pages to maintain consistency and reduce duplication. All snippets are actual `.py` or `.sh` files that developers can run directly.

## Directory Structure

```
_snippets/
├── api-examples/              # Executable API code examples
│   ├── basic_evaluate.py      # Standard evaluate() pattern
│   ├── multi_task.py          # Multiple task evaluation
│   └── result_access.py       # Accessing results
├── parameters/                # Configuration parameter examples
│   └── academic_minimal.py    # Minimal params for academic benchmarks
├── commands/                  # Executable CLI command scripts
│   └── list_tasks.sh          # Task discovery commands
└── prerequisites/             # Pre-flight check scripts
    ├── endpoint_check.py      # Endpoint health verification
    └── logprob_endpoint_check.py  # Log-probability endpoint check
```

## Usage

### In Documentation

Include snippets using MyST's `literalinclude` directive with markers:

```markdown
# In any documentation file
```{literalinclude} ../_snippets/parameters/academic_minimal.py
:language: python
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```
```

### As Standalone Scripts

All snippets are executable and can be run directly:

```bash
# Run endpoint check
export YOUR_API_KEY="your-api-key"
python docs/evaluation/_snippets/prerequisites/endpoint_check.py

# Run multi-task evaluation
python docs/evaluation/_snippets/api-examples/multi_task.py

# Run task discovery
bash docs/evaluation/_snippets/commands/list_tasks.sh
```

## Benefits

1. **Executable Examples**: All snippets are runnable code that developers can test
2. **Single Source of Truth**: Update once, reflect everywhere
3. **Consistency**: Ensure all examples use standardized patterns
4. **Testability**: Scripts can be tested in CI/CD pipelines
5. **Developer-Friendly**: Copy-paste-run workflow

## Snippet Markers

All snippets use comment markers to define the includable region:

```python
"# [snippet-start]"
# ... actual code that gets included in docs ...
"# [snippet-end]"
```

Code outside the markers (like imports, main blocks, helpers) supports standalone execution but isn't shown in documentation.

## Guidelines

When creating new snippets:

1. **Make them executable**: Include proper shebang, imports, and main blocks
2. **Use snippet markers**: Wrap the documentation-relevant code in `[snippet-start]`/`[snippet-end]`
3. **Keep them focused**: Each snippet should serve a single purpose
4. **Test before committing**: Run the script to ensure it works
5. **Add environment variable support**: Allow configuration via env vars
6. **Include helpful output**: Print success/failure messages

### Example Structure

```python
#!/usr/bin/env python3
"""
Brief description of what this script does.
"""
import os
import sys

"# [snippet-start]"
# The actual code shown in documentation
from nemo_evaluator.api.api_dataclasses import ConfigParams

params = ConfigParams(
    temperature=0.01,
    parallelism=4
)
"# [snippet-end]"

if __name__ == "__main__":
    # Additional code for standalone execution
    # Not shown in documentation
    pass
```

## Testing Snippets

Run all snippets to validate they work:

```bash
# Test endpoint checks (requires API key)
export YOUR_API_KEY="your-key"
python docs/evaluation/_snippets/prerequisites/endpoint_check.py

# Test parameter examples (syntax check)
python -c "from docs.evaluation._snippets.parameters.academic_minimal import params"

# Test shell scripts
bash docs/evaluation/_snippets/commands/list_tasks.sh --help || true
```

## Updating Snippets

When updating a snippet:

1. **Modify the .py or .sh file** directly
2. **Test the standalone script** to ensure it still works
3. **Check documentation build** to ensure literalinclude works
4. **Search for all references**: `grep -r "snippet_name" docs/`
5. **Update this README** if structure changes

