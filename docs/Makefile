# NeMo Evaluator — Fern docs convenience targets.
#
# Fern CLI root: docs/fern/ (hoisted prose lives in docs/*.mdx; nightly.yml points
# at ../../). CI workflows under .github/workflows/fern-docs-*.yml are the source
# of truth for publish/preview; these targets mirror local entry points.
#
# Sphinx (legacy) targets remain on the repo-root makefile (make docs-html, etc.).
#
# First time on this machine? Run `make docs-login` before library autodoc work.
# Evaluator prose docs do not require `fern docs md generate` unless you add
# `libraries:` to docs/fern/docs.yml later.

FERN_DIR := fern
FERN_VERSION := $(shell jq -r .version $(FERN_DIR)/fern.config.json 2>/dev/null || echo latest)
PUBLISH_WORKFLOW := Publish Fern Docs
SNAPSHOT_VERSION ?=

.DEFAULT_GOAL := help

.PHONY: help \
        docs docs-check docs-preview docs-publish docs-login \
        docs-snapshot docs-freeze \
        sphinx-html sphinx-live sphinx-publish sphinx-clean

help:
	@echo ""
	@echo "NeMo Evaluator docs/ Make targets"
	@echo "================================="
	@echo ""
	@echo "Fern docs (canonical):"
	@echo "  make docs-login        FIRST-TIME — Fern dashboard + CLI auth"
	@echo "  make docs              Start Fern dev server (localhost:3004)"
	@echo "  make docs-check        fern check + fern docs md check"
	@echo "  make docs-preview      Shared preview on *.docs.buildwithfern.com (needs DOCS_FERN_TOKEN)"
	@echo "  make docs-publish      Trigger 'Publish Fern Docs' on origin/main"
	@echo "  make docs-snapshot     Freeze docs/ into fern/versions/<ver>/pages (see below)"
	@echo ""
	@echo "Sphinx docs (legacy — repo-root makefile):"
	@echo "  make sphinx-html       Build Sphinx HTML (delegates to ../makefile)"
	@echo "  make sphinx-live       Sphinx live-reload"
	@echo "  make sphinx-publish    Sphinx publish build"
	@echo "  make sphinx-clean      Clean Sphinx _build"
	@echo ""
	@echo "Snapshot example:"
	@echo "  make docs-snapshot SNAPSHOT_VERSION=v0.2.0"
	@echo "  # then register version in fern/docs.yml, fern check, commit, tag v0.2.0"
	@echo ""

# ---------------------------------------------------------------------------
# Fern targets
# ---------------------------------------------------------------------------

docs-login:
	@echo ""
	@echo "Fern auth — one-time setup per machine"
	@echo "======================================="
	@echo ""
	@echo "  Step 1 (REQUIRED FIRST): open"
	@echo ""
	@echo "      https://dashboard.buildwithfern.com/login"
	@echo ""
	@echo "  Sign in with your @nvidia.com email (email / magic-link, not Google SSO)."
	@echo ""
	@echo "  Step 2: confirm the 'nvidia' org appears in the dashboard sidebar."
	@echo ""
	@echo "  Step 3: complete 'fern login' when prompted below (same email)."
	@echo ""
	@printf "Have you completed step 1 (dashboard sign-in)? [yes/N]: "
	@read confirm; case "$$confirm" in \
		yes|YES|y|Y) ;; \
		*) echo ""; echo "Bailing. Sign in at the dashboard URL, then re-run 'make docs-login'."; exit 1 ;; \
	esac
	@echo ""
	npx -y fern-api@$(FERN_VERSION) login

docs:
	cd $(FERN_DIR) && npx -y fern-api@$(FERN_VERSION) docs dev

docs-check:
	cd $(FERN_DIR) && npx -y fern-api@$(FERN_VERSION) check
	cd $(FERN_DIR) && npx -y fern-api@$(FERN_VERSION) docs md check

docs-preview:
	cd $(FERN_DIR) && npx -y fern-api@$(FERN_VERSION) generate --docs --preview

docs-publish:
	gh workflow run "$(PUBLISH_WORKFLOW)" --ref main

docs-snapshot docs-freeze:
	@test -n "$(SNAPSHOT_VERSION)" || { \
		echo "Usage: make docs-snapshot SNAPSHOT_VERSION=v0.2.0"; \
		exit 1; \
	}
	chmod +x $(FERN_DIR)/scripts/freeze_version.sh
	$(FERN_DIR)/scripts/freeze_version.sh "$(SNAPSHOT_VERSION)"

# ---------------------------------------------------------------------------
# Sphinx wrappers — repo-root makefile (legacy pipeline).
# ---------------------------------------------------------------------------

sphinx-html:
	$(MAKE) -C .. docs-html

sphinx-live:
	$(MAKE) -C .. docs-live

sphinx-publish:
	$(MAKE) -C .. docs-publish

sphinx-clean:
	$(MAKE) -C .. docs-clean
