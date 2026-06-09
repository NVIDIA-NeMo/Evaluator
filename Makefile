.PHONY: test test-offline test-all lint format docs docs-check docs-dev clean install pre-commit ci

# Set up a persistent editable dev environment in .venv. uv honors the
# [tool.uv] override-dependencies/conflicts that plain pip ignores.
install:
	uv venv
	uv pip install -e ".[dev]"

# The fast feedback loop: matches the pytest invocation in the OSS CI workflow.
# Excludes network/slow/e2e markers so this runs offline.
test-offline:
	uv run --extra dev pytest tests/ -m "not network and not slow and not e2e" -q --tb=short -n auto

# Default `test` target = the offline subset that CI gates on.
test: test-offline

# Full suite including network and slow tests. Run before tagging a release.
test-all:
	uv run --extra dev pytest tests/ -v

lint:
	uvx ruff check src/ tests/
	uvx ruff format --check src/ tests/

format:
	uvx ruff format src/ tests/
	uvx ruff check --fix src/ tests/

pre-commit:
	uvx pre-commit run --all-files

# Mirror the OSS CI workflow locally before pushing.
ci: lint test-offline

# Fern docs live in docs/fern (config) with MDX content under docs/. The Fern CLI
# version is pinned in docs/fern/fern.config.json; install it with `npm install -g
# fern-api@<version>`.
docs: docs-check

docs-check:
	cd docs/fern && fern check

docs-dev:
	cd docs/fern && fern docs dev

clean:
	rm -rf docs/_build eval_results dist build *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
