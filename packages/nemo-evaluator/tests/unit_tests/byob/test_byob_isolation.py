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

"""Import isolation tests for contrib.byob.

Ensures that the BYOB subpackage does not create transitive import
dependencies that could break unrelated modules (e.g. api.api_dataclasses)
if contrib.byob has a broken dependency.

See: https://github.com/NVIDIA/nemo-evaluator/pull/763 — Marta's review
comment about ensuring good separation between components.
"""

import ast
import os
import subprocess
import sys

import pytest


class TestImportIsolation:
    """Verify that contrib.byob is not transitively imported by core modules."""

    # Modules that must NEVER import contrib.byob (directly or transitively)
    PROTECTED_MODULES = [
        "nemo_evaluator.api.api_dataclasses",
        "nemo_evaluator.core.input",
        "nemo_evaluator.core.evaluate",
        "nemo_evaluator.core.entrypoint",
    ]

    @pytest.mark.parametrize("module", PROTECTED_MODULES)
    def test_core_module_importable_without_byob(self, module):
        """Importing a core module must not trigger contrib.byob import.

        Runs in a subprocess to get a clean import state.  After importing
        the target module, checks that contrib.byob is NOT in sys.modules.
        """
        code = (
            f"import {module}; "
            f"import sys; "
            f"byob_loaded = any("
            f"  k.startswith('nemo_evaluator.contrib.byob') for k in sys.modules"
            f"); "
            f"sys.exit(1 if byob_loaded else 0)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, (
            f"Importing {module} transitively loaded contrib.byob. "
            f"This breaks import isolation — a bad import in byob/__init__.py "
            f"would crash unrelated modules.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    def test_contrib_init_is_empty(self):
        """contrib/__init__.py must not import byob or any other subpackage.

        This is the structural guarantee: as long as contrib/__init__.py
        stays empty (or only has namespace boilerplate), importing
        nemo_evaluator.contrib won't pull in byob.
        """
        contrib_init = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "src",
            "nemo_evaluator",
            "contrib",
            "__init__.py",
        )
        contrib_init = os.path.abspath(contrib_init)

        if not os.path.exists(contrib_init):
            pytest.skip("contrib/__init__.py not found")

        with open(contrib_init) as f:
            content = f.read().strip()

        # Empty file or only comments/docstrings are fine
        if not content:
            return

        # Parse the AST and check for import statements
        tree = ast.parse(content)
        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        byob_imports = [
            node
            for node in imports
            if any(
                "byob" in (getattr(node, "module", "") or "")
                or "byob" in (alias.name for alias in node.names)
                for alias in getattr(node, "names", [])
            )
        ]
        assert not byob_imports, (
            "contrib/__init__.py imports byob — this breaks isolation. "
            "Remove the import to prevent transitive coupling."
        )
