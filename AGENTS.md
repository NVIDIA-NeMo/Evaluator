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
uv run pytest tests/unit_tests/ -v
```

### Launcher package

```bash
cd packages/nemo-evaluator-launcher
NEMO_EVALUATOR_LAUNCHER_LOG_DIR=/tmp .venv/bin/pytest tests/unit_tests/ -v
```

**Important**: The launcher requires `NEMO_EVALUATOR_LAUNCHER_LOG_DIR` set to a writable directory, otherwise tests fail with `ValueError: Unable to configure handler 'file'`.

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

## Git Conventions

- Branch naming: `username/feature-name`
- Commit style: conventional commits (`refactor:`, `docs:`, `fix:`, `feat:`)
- Sign-off: use `-s` flag
