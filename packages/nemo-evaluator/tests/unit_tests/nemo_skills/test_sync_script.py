# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the nemo-skills CI sync pipeline.

All tests are skipped until the CI sync pipeline (sync_skills.py) is implemented.
See Phase 6 in the project plan.
"""

import pytest

pytestmark = pytest.mark.skip(reason="CI sync pipeline not yet implemented (Phase 6)")


class TestSyncManifestParsing:
    """Tests for manifest loading and validation (T-001 through T-005)."""

    def test_t001_load_valid_manifest(self, tmp_path):
        """T-001: Load a valid manifest with all required fields."""
        pass

    def test_t002_missing_manifest_file(self):
        """T-002: Raise FileNotFoundError for nonexistent manifest."""
        pass

    def test_t003_manifest_missing_required_field(self, tmp_path):
        """T-003: Raise ValueError for manifest missing sync_dirs."""
        pass

    def test_t004_manifest_empty_sync_dirs(self, tmp_path):
        """T-004: Raise ValueError for empty sync_dirs list."""
        pass

    def test_t005_manifest_identity_rewrite_rule(self, tmp_path):
        """T-005: Raise ValueError for rewrite rule with pattern==replacement."""
        pass


class TestCopyAndRewrite:
    """Tests for file copying and import rewriting (T-006 through T-011)."""

    def test_t006_copy_files_with_include_exclude(self, tmp_path):
        """T-006: Copy files matching include patterns, excluding test_*.py."""
        pass

    def test_t007_rewrite_general_import(self, tmp_path):
        """T-007: Rewrite general nemo_skills import to _nemo_skills."""
        pass

    def test_t008_rewrite_specific_before_general(self, tmp_path):
        """T-008: Apply specific adapter rule before general rule (INV-007)."""
        pass

    def test_t009_rewrite_skip_allowlisted_import(self, tmp_path):
        """T-009: Leave allowlisted imports unchanged."""
        pass

    def test_t010_protected_file_not_overwritten(self, tmp_path):
        """T-010: Protected files remain unchanged (INV-002)."""
        pass

    def test_t011_multiple_protected_patterns(self, tmp_path):
        """T-011: All protected patterns are respected."""
        pass


class TestValidation:
    """Tests for import validation (T-012, T-013)."""

    def test_t012_validate_imports_finds_violation(self, tmp_path):
        """T-012: Detect bare nemo_skills import not in allowlist (INV-001)."""
        pass

    def test_t013_validate_imports_clean_codebase(self, tmp_path):
        """T-013: Return empty list for clean codebase (INV-001)."""
        pass


class TestEdgeCases:
    """Edge case tests for sync pipeline (T-076, T-077)."""

    def test_t076_rewrite_file_with_no_imports(self, tmp_path):
        """T-076: File with no imports returns 0 rewrites."""
        pass

    def test_t077_copy_files_empty_source(self, tmp_path):
        """T-077: Empty source directory returns CopyReport(files_copied=0)."""
        pass
