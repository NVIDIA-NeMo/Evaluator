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
    "sphinx.ext.autosummary",  #  - Added conditionally below based on package availability
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
    "sphinxcontrib.autodoc_pydantic",
    "enum_tools.autoenum",
    "sphinxcontrib.autoprogram",
    # "autodoc2.sphinx",  # Not used - apidocs/ directory not linked in navigation
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
    "apidocs/**",  # Exclude autodoc2-generated docs (not used in navigation)
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
    "ref.python",  # Expected for ambiguous cross-references (e.g., multiple 'Params' classes)
    "myst.xref_missing",  # Expected for Pydantic BaseModel docstrings that reference Pydantic's own documentation
]

# -- Options for linkcheck builder -------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-linkcheck_anchors
linkcheck_anchors = False  # Disable checking for anchors in links

# -- Options for Autodoc ---------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath("../packages/nemo-evaluator/src"))

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
autosummary_ignore_module_all = True
autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "inherited-members": False,
    "special-members": False,
    "private-members": False,
    "show-inheritance": True,
}

autodoc_pydantic_model_show_json = False
autodoc_pydantic_model_show_config_summary = True
autodoc_pydantic_model_show_field_summary = False
autodoc_pydantic_model_show_validator_summary = True


# -- Setup function to run before building docs --------------------------------
def setup(app):
    """Sphinx setup function that runs before building documentation.

    This function is called by Sphinx during the build process and can be used
    to run pre-build tasks like regenerating autogenerated documentation.
    """
    import pathlib
    import subprocess

    from sphinx.util import logging as sphinx_logging

    logger = sphinx_logging.getLogger(__name__)

    # Get the docs directory (where conf.py is located)
    docs_dir = pathlib.Path(__file__).parent.resolve()
    repo_root = docs_dir.parent

    # Path to autogen script
    autogen_script = (
        repo_root
        / "packages"
        / "nemo-evaluator-launcher"
        / "scripts"
        / "autogen_task_yamls.py"
    )

    # Expected output files that should be generated
    expected_harnesses_dir = docs_dir / "task_catalog" / "harnesses"
    expected_benchmarks_table_file = docs_dir / "task_catalog" / "benchmarks-table.md"

    # Only run if script exists
    if autogen_script.exists():
        try:
            logger.info("=" * 80)
            logger.info("Running autogen script to regenerate task documentation...")
            logger.info("=" * 80)

            # Run autogen script to regenerate task documentation
            # Use the same Python interpreter that's running Sphinx
            # Add launcher package to Python path so script can import its modules
            launcher_src = repo_root / "packages" / "nemo-evaluator-launcher" / "src"
            evaluator_src = repo_root / "packages" / "nemo-evaluator" / "src"

            # Build PYTHONPATH with both launcher and evaluator packages
            env = os.environ.copy()
            pythonpath_parts = []
            if launcher_src.exists():
                pythonpath_parts.append(str(launcher_src))
            if evaluator_src.exists():
                pythonpath_parts.append(str(evaluator_src))

            existing_pythonpath = env.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_parts.append(existing_pythonpath)

            if pythonpath_parts:
                env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

            # Run autogen script - FAIL BUILD if it fails
            # Stream stdout in real-time for better UX, but capture stderr for error reporting
            result = subprocess.run(
                [sys.executable, str(autogen_script)],
                cwd=str(repo_root),
                env=env,
                stdout=None,  # Let stdout stream normally so users see progress
                stderr=subprocess.PIPE,  # Capture stderr for error reporting
                text=True,
                check=False,  # We'll handle the error ourselves for better messaging
            )

            if result.returncode == 0:
                logger.info("✓ Autogen script completed successfully")

                # Verify expected output files exist
                missing_files = []
                if not expected_harnesses_dir.exists():
                    missing_files.append(str(expected_harnesses_dir))
                if not expected_benchmarks_table_file.exists():
                    missing_files.append(str(expected_benchmarks_table_file))

                if missing_files:
                    error_msg = (
                        "ERROR: Autogen script completed but expected output files are missing:\n"
                        + "\n".join(f"  - {f}" for f in missing_files)
                        + "\n\n"
                        + "This indicates the autogen script did not generate the expected documentation files.\n"
                        + "Please check the autogen script output above for errors."
                    )
                    logger.error("=" * 80)
                    logger.error(error_msg)
                    logger.error("=" * 80)
                    raise RuntimeError(error_msg)
            else:
                # Build detailed error message
                error_msg_parts = [
                    "=" * 80,
                    "ERROR: Autogen script failed with exit code {}".format(
                        result.returncode
                    ),
                    "=" * 80,
                    "",
                    "The documentation build cannot proceed because task documentation generation failed.",
                    "",
                    "STDERR:",
                    "-" * 80,
                ]
                if result.stderr:
                    error_msg_parts.extend(result.stderr.strip().split("\n"))
                else:
                    error_msg_parts.append(
                        "(no stderr output - check stdout above for details)"
                    )

                error_msg_parts.extend(
                    [
                        "",
                        "=" * 80,
                        "To fix this issue:",
                        "1. Ensure all_tasks_irs.yaml is up to date by running:",
                        "   python packages/nemo-evaluator-launcher/scripts/load_framework_definitions.py",
                        "2. Check that all required dependencies are installed",
                        "3. Verify the autogen script can run independently",
                        "=" * 80,
                    ]
                )

                error_msg = "\n".join(error_msg_parts)

                # Log error prominently
                logger.error(error_msg)
                raise RuntimeError(
                    f"Autogen script failed with exit code {result.returncode}. "
                    "See error output above for details."
                )
        except subprocess.CalledProcessError as e:
            # This shouldn't happen with check=False, but handle it just in case
            error_msg = (
                f"ERROR: Autogen script execution failed: {e}\n"
                f"Command: {e.cmd}\n"
                f"Return code: {e.returncode}\n"
            )
            if e.stdout:
                error_msg += f"STDOUT:\n{e.stdout}\n"
            if e.stderr:
                error_msg += f"STDERR:\n{e.stderr}\n"
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise RuntimeError(
                "Autogen script execution failed. See error output above."
            ) from e
        except Exception as e:
            # Log error prominently and fail the build
            import traceback

            error_msg = (
                f"ERROR: Unexpected error while running autogen script: {e}\n"
                f"\nTraceback:\n{traceback.format_exc()}"
            )
            logger.error("=" * 80)
            logger.error(error_msg)
            logger.error("=" * 80)
            raise RuntimeError(
                "Autogen script execution failed. See error output above."
            ) from e
    else:
        # Script not found - this is okay if launcher package isn't available
        logger.info(
            "ℹ Autogen script not found, skipping task documentation regeneration"
        )

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
