# Makefile targets for Sphinx documentation (all targets prefixed with 'docs-')
# Adapted for NeMo Evaluator project structure with pure uv dependency management

.PHONY: docs-html docs-clean docs-live docs-env docs-publish \
        docs-html-internal docs-html-ga docs-html-ea docs-html-draft \
        docs-live-internal docs-live-ga docs-live-ea docs-live-draft \
        docs-publish-internal docs-publish-ga docs-publish-ea docs-publish-draft

# Usage:
#   make docs-html DOCS_ENV=internal   # Build docs for internal use
#   make docs-html DOCS_ENV=ga         # Build docs for GA
#   make docs-html                     # Build docs with no special tag
#   make docs-live DOCS_ENV=draft      # Live server with draft tag
#   make docs-publish DOCS_ENV=ga      # Production build (fails on warnings)

DOCS_ENV ?=

# Detect OS for cross-platform compatibility
ifeq ($(OS),Windows_NT)
    RM_CMD = if exist docs\_build rmdir /s /q docs\_build
    PKG_DIR = packages\nemo-evaluator
    DOCS_DIR = ..\..\docs
    BUILD_DIR = ..\..\docs\_build\html
else
    RM_CMD = cd docs && rm -rf _build
    PKG_DIR = packages/nemo-evaluator
    DOCS_DIR = ../../docs
    BUILD_DIR = ../../docs/_build/html
endif

# Main documentation targets using uv run

docs-html:
	@echo "Building HTML documentation..."
	cd $(PKG_DIR) && uv run --group docs sphinx-build -b html $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-publish:
	@echo "Building HTML documentation for publication (fail on warnings)..."
	cd $(PKG_DIR) && uv run --group docs sphinx-build --fail-on-warning --builder html $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-clean:
	@echo "Cleaning built documentation..."
	$(RM_CMD)

docs-live:
	@echo "Starting live-reload server (sphinx-autobuild)..."
	cd $(PKG_DIR) && uv run --group docs sphinx-autobuild $(if $(DOCS_ENV),-t $(DOCS_ENV)) $(DOCS_DIR) $(BUILD_DIR)

docs-env:
	@echo "Syncing documentation dependencies..."
	cd $(PKG_DIR) && uv sync --group docs
	@echo "Documentation dependencies synced!"
	@echo "You can now run 'make docs-html' or 'make docs-live'"

# HTML build shortcuts

docs-html-internal:
	$(MAKE) docs-html DOCS_ENV=internal

docs-html-ga:
	$(MAKE) docs-html DOCS_ENV=ga

docs-html-ea:
	$(MAKE) docs-html DOCS_ENV=ea

docs-html-draft:
	$(MAKE) docs-html DOCS_ENV=draft

# Publish build shortcuts

docs-publish-internal:
	$(MAKE) docs-publish DOCS_ENV=internal

docs-publish-ga:
	$(MAKE) docs-publish DOCS_ENV=ga

docs-publish-ea:
	$(MAKE) docs-publish DOCS_ENV=ea

docs-publish-draft:
	$(MAKE) docs-publish DOCS_ENV=draft

# Live server shortcuts

docs-live-internal:
	$(MAKE) docs-live DOCS_ENV=internal

docs-live-ga:
	$(MAKE) docs-live DOCS_ENV=ga

docs-live-ea:
	$(MAKE) docs-live DOCS_ENV=ea

docs-live-draft:
	$(MAKE) docs-live DOCS_ENV=draft

# Additional convenience targets

docs-help:
	@echo "Available documentation targets:"
	@echo "  docs-env        - Sync documentation dependencies with uv"
	@echo "  docs-html       - Build HTML documentation"
	@echo "  docs-live       - Start live-reload server for development"
	@echo "  docs-publish    - Build documentation with strict error checking"
	@echo "  docs-clean      - Clean built documentation"
	@echo ""
	@echo "Environment-specific targets (replace 'html' with 'live' or 'publish'):"
	@echo "  docs-html-internal  - Build with 'internal' tag"
	@echo "  docs-html-ga        - Build with 'ga' tag"
	@echo "  docs-html-ea        - Build with 'ea' tag"
	@echo "  docs-html-draft     - Build with 'draft' tag"
	@echo ""
	@echo "Usage examples:"
	@echo "  make docs-env                    # Sync dependencies first (recommended)"
	@echo "  make docs-html                   # Basic build"
	@echo "  make docs-html DOCS_ENV=ga       # Build with GA tag"
	@echo "  make docs-live DOCS_ENV=draft    # Live server with draft tag"
	@echo ""
	@echo "Note: Uses 'uv run' with dependencies from packages/nemo-evaluator/pyproject.toml"