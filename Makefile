.PHONY: test test-offline test-all lint format docs clean install pre-commit ci

# Match what `pip install -e ".[dev]"` does in CI.
install:
	pip install -e ".[dev]"

# The fast feedback loop: matches the pytest invocation in the OSS CI workflow.
# Excludes network/slow/e2e markers so this runs offline.
test-offline:
	pytest tests/ -m "not network and not slow and not e2e" -q --tb=short -n auto

# Default `test` target = the offline subset that CI gates on.
test: test-offline

# Full suite including network and slow tests. Run before tagging a release.
test-all:
	pytest tests/ -v

lint:
	ruff check src/ tests/
	ruff format --check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

pre-commit:
	pre-commit run --all-files

# Mirror the OSS CI workflow locally before pushing.
ci: lint test-offline

docs:
	sphinx-build -b html docs docs/_build/html

clean:
	rm -rf docs/_build eval_results dist build *.egg-info .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
