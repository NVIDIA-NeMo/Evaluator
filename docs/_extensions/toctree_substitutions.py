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
Custom extension to enable MyST substitutions in toctree entries.
"""

import re

from sphinx.application import Sphinx
from sphinx.directives.other import TocTree


class SubstitutionTocTree(TocTree):
    """Enhanced TocTree that supports MyST substitutions."""

    def run(self):
        # Get MyST substitutions from config
        substitutions = self.state.document.settings.env.config.myst_substitutions or {}

        # Process each line in the content
        processed_content = []
        for line in self.content:
            processed_line = line
            # Replace substitutions using regex pattern
            for key, value in substitutions.items():
                pattern = r"\{\{\s*" + re.escape(key) + r"\s*\}\}"
                processed_line = re.sub(pattern, value, processed_line)
            processed_content.append(processed_line)

        # Update content with processed lines
        self.content = processed_content

        # Call parent implementation
        return super().run()


def setup(app: Sphinx):
    """Setup function for the extension."""
    # Override the default toctree directive
    app.add_directive("toctree", SubstitutionTocTree, override=True)

    return {
        "version": "0.1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
