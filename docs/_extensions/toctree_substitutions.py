"""
Custom extension to enable MyST substitutions in toctree entries.
"""

import re
from docutils import nodes
from sphinx.application import Sphinx
from sphinx.parsers.rst import directives
from sphinx.directives.other import TocTree
from myst_parser.myst_parser import MystParser


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
                pattern = r'\{\{\s*' + re.escape(key) + r'\s*\}\}'
                processed_line = re.sub(pattern, value, processed_line)
            processed_content.append(processed_line)
        
        # Update content with processed lines
        self.content = processed_content
        
        # Call parent implementation
        return super().run()


def setup(app: Sphinx):
    """Setup function for the extension."""
    # Override the default toctree directive
    app.add_directive('toctree', SubstitutionTocTree, override=True)
    
    return {
        'version': '0.1.0',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
