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

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys

# Add custom extensions directory to Python path
sys.path.insert(0, os.path.abspath("_extensions"))

project = "NeMo Evaluator SDK"
copyright = "2025, NVIDIA Corporation"
author = "NVIDIA Corporation"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    # "autodoc2" - Added conditionally below based on package availability
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks,
    "sphinx_design",  # For grid layout
    "sphinx.ext.ifconfig",  # For conditional content
    "content_gating",  # Unified content gating extension
    "myst_codeblock_substitutions",  # Our custom MyST substitutions in code blocks
    "json_output",  # Generate JSON output for each page
    "search_assets",  # Enhanced search assets extension
    # "ai_assistant",  # AI Assistant extension for intelligent search responses
    "swagger_plugin_for_sphinx",  # For Swagger API documentation
    "sphinxcontrib.mermaid",  # For Mermaid diagrams
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "_extensions/*/README.md",  # Exclude README files in extension directories
    "_extensions/README.md",  # Exclude main extensions README
    "_extensions/*/__pycache__",  # Exclude Python cache directories
    "_extensions/*/*/__pycache__",  # Exclude nested Python cache directories
]

# -- Options for Intersphinx -------------------------------------------------
# Cross-references to external NVIDIA documentation
intersphinx_mapping = {
    "ctk": (
        "https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest",
        None,
    ),
    "gpu-op": (
        "https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest",
        None,
    ),
    "ngr-tk": ("https://docs.nvidia.com/nemo/guardrails/latest", None),
    "nim-cs": (
        "https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-contentsafety/latest/",
        None,
    ),
    "nim-tc": (
        "https://docs.nvidia.com/nim/llama-3-1-nemoguard-8b-topiccontrol/latest/",
        None,
    ),
    "nim-jd": ("https://docs.nvidia.com/nim/nemoguard-jailbreakdetect/latest/", None),
    "nim-llm": ("https://docs.nvidia.com/nim/large-language-models/latest/", None),
    "driver-linux": (
        "https://docs.nvidia.com/datacenter/tesla/driver-installation-guide",
        None,
    ),
    "nim-op": ("https://docs.nvidia.com/nim-operator/latest", None),
}

# Intersphinx timeout for slow connections
intersphinx_timeout = 30

# -- Options for JSON Output -------------------------------------------------
# Configure the JSON output extension for comprehensive search indexes
json_output_settings = {
    "enabled": True,
}

# -- Options for AI Assistant -------------------------------------------------
# Configure the AI Assistant extension for intelligent search responses
ai_assistant_enabled = True
ai_assistant_endpoint = "<your endpoint here"
ai_assistant_api_key = ""  # Set this to your Pinecone API key
ai_trigger_threshold = 2  # Trigger AI when fewer than N search results
ai_auto_trigger = True  # Automatically trigger AI analysis

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
    "attrs_inline",  # Enables inline attributes for markdown
    "substitution",  # Enables substitution for markdown
]

myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# MyST substitutions for reusable variables across documentation
myst_substitutions = {
    "product_name": "NVIDIA NeMo Evaluator",
    "product_name_short": "NeMo Evaluator",
    "company": "NVIDIA",
    "version": release,
    "current_year": "2025",
    "github_repo": "https://github.com/NVIDIA-NeMo/Evaluator",
    "docs_url": "https://docs.nvidia.com/nemo/evaluator/latest/index.html",
    "support_email": "update-me",
    "min_python_version": "3.8",
    "recommended_cuda": "12.0+",
    "docker_compose_latest": "25.10",
}

# Enable figure numbering
numfig = True

# Optional: customize numbering format
numfig_format = {"figure": "Figure %s", "table": "Table %s", "code-block": "Listing %s"}

# Optional: number within sections
numfig_secnum_depth = 1  # Gives you "Figure 1.1, 1.2, 2.1, etc."


# Suppress expected warnings for conditional content builds
suppress_warnings = [
    "toc.not_included",  # Expected when video docs are excluded from GA builds
    "toc.no_title",  # Expected for helm docs that include external README files
    "docutils",  # Expected for autodoc2-generated content with regex patterns and complex syntax
    "ref.python",  # Expected for ambiguous autodoc2 cross-references (e.g., multiple 'Params' classes)
    "myst.xref_missing",  # Expected for Pydantic BaseModel docstrings that reference Pydantic's own documentation
]

# -- Options for linkcheck builder -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-linkcheck_anchors
linkcheck_anchors = False  # Disable checking for anchors in links

# -- Options for Autodoc2 ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../packages/nemo-evaluator/src"))

# Conditional autodoc2 configuration - only enable if packages exist
# Note: We point to the parent package rather than individual subpackages because
# the subpackages have relative imports between them (e.g., api imports from core)
autodoc2_packages_list = ["../packages/nemo-evaluator/src/nemo_evaluator/"]

# Check if any of the packages actually exist before enabling autodoc2
autodoc2_packages = []
for pkg_path in autodoc2_packages_list:
    abs_pkg_path = os.path.abspath(os.path.join(os.path.dirname(__file__), pkg_path))
    if os.path.exists(abs_pkg_path):
        autodoc2_packages.append(pkg_path)

# Only include autodoc2 in extensions if we have valid packages
if autodoc2_packages:
    if "autodoc2" not in extensions:
        extensions.append("autodoc2")

    autodoc2_render_plugin = "myst"  # Use MyST for rendering docstrings
    autodoc2_output_dir = "apidocs"  # Output directory for autodoc2 (relative to docs/)

    # ==================== GOOD DEFAULTS FOR CLEANER DOCS ====================

    # Hide implementation details - good defaults for cleaner docs
    autodoc2_hidden_objects = [
        "dunder",  # Hide __methods__ like __init__, __str__, etc.
        "private",  # Hide _private methods and variables
        "inherited",  # Hide inherited methods to reduce clutter
    ]

    # Enable module summaries for better organization
    autodoc2_module_summary = True

    # Sort by name for consistent organization
    autodoc2_sort_names = True

    # Enhanced docstring processing for better formatting
    autodoc2_docstrings = "all"  # Include all docstrings for comprehensive docs

    # Include class inheritance information - useful for users
    autodoc2_class_inheritance = True

    # Handle class docstrings properly (merge __init__ with class doc)
    autodoc2_class_docstring = "merge"

    # Better type annotation handling
    autodoc2_type_guard_imports = True

    # Replace common type annotations for better readability
    autodoc2_replace_annotations = [
        ("typing.Union", "Union"),
        ("typing.Optional", "Optional"),
        ("typing.List", "List"),
        ("typing.Dict", "Dict"),
        ("typing.Any", "Any"),
    ]

    # Don't require __all__ to be defined - document all public members
    autodoc2_module_all_regexes = []  # Empty list means don't require __all__

    # Skip common test and internal modules - customize for your project
    autodoc2_skip_module_regexes = [
        r".*\.tests?.*",  # Skip test modules
        r".*\.test_.*",  # Skip test files
        r".*\._.*",  # Skip private modules
        r".*\.conftest",  # Skip conftest files
    ]

    # Load index template from external file for better maintainability
    template_path = os.path.join(
        os.path.dirname(__file__), "_templates", "autodoc2_index.rst"
    )
    with open(template_path) as f:
        autodoc2_index_template = f.read()

    # This is a workaround that uses the parser located in autodoc2_docstrings_parser.py to allow autodoc2 to
    # render google style docstrings.
    # Related Issue: https://github.com/sphinx-extensions2/sphinx-autodoc2/issues/33
    # autodoc2_docstring_parser_regexes = [
    #     (r".*", "docs.autodoc2_docstrings_parser"),
    # ]
else:
    # Remove autodoc2 from extensions if no valid packages
    if "autodoc2" in extensions:
        extensions.remove("autodoc2")
    print("INFO: autodoc2 disabled - no valid packages found in autodoc2_packages_list")

# -- Options for Napoleon (Google Style Docstrings) -------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Focus on Google style only
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"

html_theme_options = {
    "switcher": {
        "json_url": "./versions1.json",
        "version_match": release,
    },
    # Configure PyData theme search
    "search_bar_text": "Search NVIDIA docs...",
    "navbar_persistent": ["search-button"],  # Ensure search button is present
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}

# Add our static files directory
# html_static_path = ["_static"]

html_extra_path = ["project.json", "versions1.json"]

# Note: JSON output configuration has been moved to the consolidated
# json_output_settings dictionary above for better organization and new features!
