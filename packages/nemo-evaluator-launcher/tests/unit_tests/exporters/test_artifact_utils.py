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

"""Tests for artifact utilities in exporters.utils."""

import tempfile
from pathlib import Path

from nemo_evaluator_launcher.exporters.utils import (
    OPTIONAL_ARTIFACTS,
    REQUIRED_ARTIFACTS,
    get_available_artifacts,
    get_relevant_artifacts,
    validate_artifacts,
)


class TestArtifactUtils:
    def test_get_relevant_artifacts(self):
        all_artifacts = get_relevant_artifacts()
        expected = REQUIRED_ARTIFACTS + OPTIONAL_ARTIFACTS
        assert all_artifacts == expected
        assert "results.yml" in all_artifacts
        assert "eval_factory_metrics.json" in all_artifacts
        assert "omni-info.json" in all_artifacts

    def test_validate_artifacts_missing_dir(self):
        result = validate_artifacts(Path("/nonexistent"))
        assert result["can_export"] is False
        assert result["missing_required"] == REQUIRED_ARTIFACTS
        assert result["missing_optional"] == OPTIONAL_ARTIFACTS
        assert "not found" in result["message"]

    def test_validate_artifacts_all_present(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            for artifact in get_relevant_artifacts():
                (artifacts_dir / artifact).touch()
            result = validate_artifacts(artifacts_dir)
            assert result["can_export"] is True
            assert result["missing_required"] == []
            assert result["missing_optional"] == []
            assert "All artifacts available" in result["message"]

    def test_get_available_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifacts_dir = Path(tmpdir)
            (artifacts_dir / "results.yml").touch()
            (artifacts_dir / "omni-info.json").touch()
            available = get_available_artifacts(artifacts_dir)
            assert "results.yml" in available
            assert "omni-info.json" in available
            assert "eval_factory_metrics.json" not in available
