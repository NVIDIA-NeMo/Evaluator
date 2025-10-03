# Get Started Snippets

This directory contains **executable** code snippets for the get-started documentation section. All snippets are actual `.py` or `.sh` files that can be run directly or included in documentation.

## Directory Structure

```
_snippets/
├── install_launcher.sh         # Install launcher with exporters
├── install_core.sh             # Install core library  
├── install_containers.sh       # Pull NGC containers
├── verify_launcher.sh          # Verify launcher installation
├── verify_core.py              # Verify core installation
├── launcher_basic.sh           # Basic launcher usage
├── launcher_full_example.sh    # Complete launcher workflow
├── core_basic.py               # Basic Python API
├── core_full_example.py        # Complete Python API workflow
├── core_multi_benchmark.py     # Multi-benchmark evaluation
└── container_run.sh            # Direct container execution
```

## Usage

### In Documentation

Include snippets using MyST's `literalinclude` directive:

```markdown
```{literalinclude} ../_snippets/install_launcher.sh
:language: bash
:start-after: "# [snippet-start]"
:end-before: "# [snippet-end]"
```
```

### As Standalone Scripts

All snippets are executable:

```bash
# Run installation
bash docs/get-started/_snippets/install_launcher.sh

# Run quickstart
bash docs/get-started/_snippets/launcher_basic.sh

# Run verification
python docs/get-started/_snippets/verify_core.py
```

## Snippet Markers

All snippets use comment markers to define the includable region:

```bash
"# [snippet-start]"
# ... actual code shown in docs ...
"# [snippet-end]"
```

Code outside markers supports standalone execution but isn't shown in documentation.

## Validation

Validate all snippets using the validation script:

```bash
# From repository root
python scripts/validate_doc_snippets.py --verbose
```

This checks:
- ✓ Syntax correctness
- ✓ Import validity  
- ✓ API usage accuracy
- ✓ Code quality (linting)

## Environment Variables

Many scripts require environment variables:

```bash
# For NVIDIA Build API
export NGC_API_KEY="your-api-key-here"

# For custom endpoints
export MY_API_KEY="your-api-key"

# For container versions
export DOCKER_TAG="25.08.1"
```

## Testing Snippets

```bash
# Set required environment variables
export NGC_API_KEY="your-key"

# Test shell scripts (syntax only, won't actually install)
bash -n docs/get-started/_snippets/installation/*.sh

# Test Python scripts (imports only)
python -m py_compile docs/get-started/_snippets/**/*.py

# Test verification scripts
python docs/get-started/_snippets/verification/verify_core.py
```

## Guidelines

When creating new snippets:

1. **Make them executable**: Include proper shebang and imports
2. **Use snippet markers**: Wrap docs-relevant code in `[snippet-start]`/`[snippet-end]`
3. **Keep them focused**: Single purpose per snippet
4. **Test before committing**: Run to ensure it works
5. **Support env vars**: Allow configuration via environment variables
6. **Add helpful output**: Print success/failure messages

