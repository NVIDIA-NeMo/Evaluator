#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Freeze the hoisted docs/ tree into docs/fern/versions/<VERSION>/pages/ and
# generate versions/<VERSION>.yml from nightly.yml (../../ → ./<VERSION>/pages/).
#
# Usage (from repo root or docs/fern):
#   ./scripts/freeze_version.sh v0.2.0
#
# After running, register the version in docs/fern/docs.yml if new, add redirects
# for /nemo/evaluator/<VERSION>/…, and commit. Tag the release commit as
#   git tag <VERSION> && git push origin <VERSION>
# so publish-fern-docs.yml can optionally extract tag content at publish time.

set -euo pipefail

VERSION="${1:-}"
if [[ -z "$VERSION" ]]; then
  echo "Usage: $0 <version>" >&2
  echo "Example: $0 v0.2.0" >&2
  exit 1
fi

if [[ ! "$VERSION" =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
  echo "Warning: expected semver tag like v0.1.0 (got: $VERSION)" >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FERN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DOCS_DIR="$(cd "$FERN_DIR/.." && pwd)"
PAGES_DIR="$FERN_DIR/versions/$VERSION/pages"
NIGHTLY_YML="$FERN_DIR/versions/nightly.yml"
VERSION_YML="$FERN_DIR/versions/$VERSION.yml"

if [[ ! -f "$NIGHTLY_YML" ]]; then
  echo "Error: missing $NIGHTLY_YML" >&2
  exit 1
fi

echo "Freezing $DOCS_DIR -> $PAGES_DIR"
rm -rf "$PAGES_DIR"
mkdir -p "$PAGES_DIR"

rsync -a \
  --exclude 'fern/' \
  --exclude '_extensions/' \
  --exclude '_build/' \
  --exclude '_static/' \
  --exclude '_templates/' \
  --exclude 'apidocs/' \
  --exclude '.DS_Store' \
  --exclude '__pycache__/' \
  --exclude '*.rst' \
  --exclude 'conf.py' \
  --exclude 'versions1.json' \
  --exclude 'project.json' \
  --exclude 'pyproject.toml' \
  --exclude 'uv.lock' \
  --exclude 'broken_links_false_positives.json' \
  --exclude 'Makefile' \
  "$DOCS_DIR/" "$PAGES_DIR/"

sed 's|../../|./'"$VERSION"'/pages/|g' "$NIGHTLY_YML" > "$VERSION_YML"

MDX_COUNT="$(find "$PAGES_DIR" -name '*.mdx' | wc -l | tr -d ' ')"
echo "Wrote $VERSION_YML"
echo "Copied $MDX_COUNT MDX files under $PAGES_DIR"

if grep -q "path: versions/$VERSION.yml" "$FERN_DIR/docs.yml" 2>/dev/null; then
  echo "docs.yml already lists versions/$VERSION.yml"
else
  echo ""
  echo "Next: add to docs/fern/docs.yml under versions:"
  echo "  - display-name: \"${VERSION#v}\""
  echo "    path: versions/$VERSION.yml"
  echo "    slug: $VERSION"
  echo "    availability: stable   # optional, for GA/stable badge"
  echo ""
  echo "Add redirects for /nemo/evaluator/$VERSION/… (see existing v0.1.0 rules)."
  echo "Run: cd docs/fern && fern check && fern docs md check"
fi
