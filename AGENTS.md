# Agents Guide

## Repository Structure

Monorepo with two Python packages under `packages/`:

- `packages/nemo-evaluator` — Core evaluator SDK
- `packages/nemo-evaluator-launcher` — Launcher that depends on core

Each package has its own `pyproject.toml`, `uv.lock`, and `.venv`.

## Package Manager

This project uses **uv**. Key commands:

```bash
uv sync              # Install dependencies
uv run <command>     # Run command in package venv
uv pip install -e <path>  # Editable install
```

## Running Tests

### Core package

```bash
cd packages/nemo-evaluator
uv run --group test pytest tests/unit_tests/ -v
```

### Launcher package

```bash
cd packages/nemo-evaluator-launcher
uv run --group dev pytest tests/unit_tests/ -v
```

## Pre-commit

See `CONTRIBUTING.md`. Run from each package directory:

```bash
cd packages/nemo-evaluator && uv run pre-commit run --all-files
cd packages/nemo-evaluator-launcher && uv run pre-commit run --all-files
```

Uses **ruff** for linting and formatting. Ruff may auto-fix files in place — re-run tests afterwards.

## Cross-Package Development

The launcher depends on `nemo-evaluator` from PyPI. When making local changes to the core package that the launcher needs:

```bash
cd packages/nemo-evaluator-launcher
uv pip install -e ../nemo-evaluator
```

Without this, new or modified modules in the core package won't be visible to the launcher.

## Logging

Prefer `logger` over `print()` for all output. Use `print()` only when raw unformatted output is specifically needed (e.g. `--dry-run` dumping YAML to stdout).

This project uses **structlog**-style loggers. Do **not** use `%s` positional args — they won't be interpolated. Use f-strings for the message and keyword args for structured data:

```python
# Correct
logger.info(f"Processing {name}", count=10, path=str(file))

# Wrong — %s will appear literally in the output
logger.info("Processing %s", name)
```

## Re-exports

Do not add re-exports (importing a symbol from one module and exporting it from another) unless you have confirmed it is needed for backward compatibility.

## Git Conventions

- Branch naming: `username/feature-name`
- Commit style: conventional commits (`refactor:`, `docs:`, `fix:`, `feat:`)
- Sign-off: use `-s` flag
