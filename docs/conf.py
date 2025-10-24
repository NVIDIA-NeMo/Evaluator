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
    "sphinx.ext.autodoc",  # Standard autodoc - required by autodoc_pydantic
    "sphinx.ext.autosummary",  # Auto-generate API docs from docstrings (like autodoc2)
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinxcontrib.autodoc_pydantic",  # Specialized support for Pydantic models
    "autogen_api_docs",  # Custom: Auto-generate API docs with Pydantic detection
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
    # Note: keeping apidocs/ active since autogen extension is disabled
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
    "docker_compose_latest": "25.09",
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
    "ref.python",  # Expected for ambiguous cross-references (e.g., multiple 'Params' classes)
    "myst.xref_missing",  # Expected for Pydantic BaseModel docstrings that reference Pydantic's own documentation
    "autodoc.import_object",  # Expected when optional dependencies are not installed
]

# -- Options for Autodoc (Standard) and Pydantic ----------------------------
# Add package paths for autodoc to find modules
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../packages/nemo-evaluator/src"))

# Standard autodoc settings
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",  # Keep source order
    "special-members": False,
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "show-inheritance": True,
}

# Type hints configuration
autodoc_typehints = "description"  # Show type hints in description, not signature
autodoc_typehints_description_target = "documented"
autodoc_type_aliases = {
    "Optional": "typing.Optional",
    "Union": "typing.Union",
    "List": "typing.List",
    "Dict": "typing.Dict",
}

# ==================== AUTODOC_PYDANTIC CONFIGURATION ====================
# Purpose-built for Pydantic models - extracts Field descriptions properly

# Model documentation settings
autodoc_pydantic_model_show_json = False  # Don't show JSON schema by default
autodoc_pydantic_model_show_config_summary = False  # Don't show model_config details
autodoc_pydantic_model_show_config_member = False  # Don't show model_config as member
autodoc_pydantic_model_show_validator_summary = False  # Disable to avoid ordering errors
autodoc_pydantic_model_show_validator_members = False  # Disable to avoid ordering errors
autodoc_pydantic_model_summary_list_order = "alphabetical"  # Use alphabetical
autodoc_pydantic_model_member_order = "alphabetical"  # Use alphabetical to avoid ordering errors
autodoc_pydantic_model_undoc_members = True  # Include members without docstrings
autodoc_pydantic_model_hide_paramlist = False  # Show __init__ parameters

# Field documentation settings - CRITICAL FOR YOUR USE CASE
autodoc_pydantic_field_list_validators = False  # Disable to avoid ordering errors with nested classes
autodoc_pydantic_field_doc_policy = "description"  # Use Field description as docstring
autodoc_pydantic_field_show_alias = False  # Don't show field aliases (can cause ordering issues)
autodoc_pydantic_field_show_default = True  # Show default values
autodoc_pydantic_field_show_required = True  # Mark required fields
autodoc_pydantic_field_show_constraints = True  # Show field constraints (gt, lt, etc.)
autodoc_pydantic_field_signature_prefix = "field"  # Prefix for field signatures
autodoc_pydantic_field_swap_name_and_alias = False  # Keep field names as-is

# Settings documentation
autodoc_pydantic_settings_show_json = False
autodoc_pydantic_settings_show_config_summary = False
autodoc_pydantic_settings_show_validator_summary = True
autodoc_pydantic_settings_show_validator_members = True
autodoc_pydantic_settings_signature_prefix = "setting"

# Validator documentation
autodoc_pydantic_validator_signature_prefix = "validator"
autodoc_pydantic_validator_replace_signature = True
autodoc_pydantic_validator_list_fields = True  # Show which fields validators apply to

# Config documentation
autodoc_pydantic_config_signature_prefix = "config"
autodoc_pydantic_config_members = True

# ==================== AUTOSUMMARY CONFIGURATION ====================
# Auto-generation similar to autodoc2 - scans packages and creates doc files

# Enable automatic generation of stub pages from autosummary directives
autosummary_generate = True

# Generate documentation even for items that don't have docstrings
autosummary_generate_overwrite = True

# Use custom templates that handle Pydantic models
autosummary_imported_members = False  # Don't document imported members

# Mock imports that might not be available during doc build
autodoc_mock_imports = []

# List of packages to document (similar to autodoc2_packages)
# These will be used in the API reference index
nemo_evaluator_packages = [
    "nemo_evaluator.api",
    "nemo_evaluator.core", 
    "nemo_evaluator.adapters",
    "nemo_evaluator.logging",
    "nemo_evaluator.cli",
]

# Autosummary settings for better output
autosummary_ignore_module_all = False  # Respect __all__ if present

# ==================== AUTO-GENERATION CONFIGURATION ====================
# Similar to autodoc2 - automatically generate API docs with Pydantic detection

# Output directory for auto-generated API docs (relative to docs/)
autogen_api_output_dir = "apidocs"  # Replaces old autodoc2 location

# Template file for the API index page (in _templates/)
autogen_api_template = "autogen_api_index.rst"

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
