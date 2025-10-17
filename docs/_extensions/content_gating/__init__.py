# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Content Gating Extension for Sphinx

Provides conditional content rendering based on release stage tags.
Supports filtering at multiple levels:
- Document level (via frontmatter)
- Toctree level (global and per-entry)
- Grid card level

Usage:
- Add tags during build: sphinx-build -t ga docs/ _build/
- Use :only: conditions in directives and frontmatter
- Supports conditions like 'ga', 'not ga', 'ea', 'not ea', 'internal', 'not internal'
"""

from sphinx.application import Sphinx

from .conditional_directives import setup as setup_conditional_directives
from .document_filter import setup as setup_document_filter


def setup(app: Sphinx):
    """
    Setup function for the content gating extension.
    """
    # Setup document-level filtering
    setup_document_filter(app)

    # Setup conditional directives (toctree and grid-item-card)
    setup_conditional_directives(app)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
