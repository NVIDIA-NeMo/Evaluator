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
Custom Sphinx extension to enable MyST substitutions in standard code blocks.

This extension pre-processes MyST markdown files to replace {{ variable }} substitutions
inside standard ``` code blocks before MyST parses the content.

Usage in any .md file:
```bash
helm install my-release oci://nvcr.io/nvidia/nemo-curator --version {{ version }}
kubectl get pods -n {{ product_name_short }}-namespace
```

The substitutions will be replaced with their values from myst_substitutions in conf.py.
"""

import re

from sphinx.application import Sphinx
from sphinx.util import logging

logger = logging.getLogger(__name__)


def process_myst_source(app, docname, source):
    """
    Process MyST source files to handle substitutions in code blocks.

    This is called by Sphinx's 'source-read' event for each document.
    """
    # Get substitutions from config
    substitutions = getattr(app.config, "myst_substitutions", {})

    if not substitutions:
        return

    # Process the source content
    original_content = source[0]
    processed_content = process_codeblock_substitutions(original_content, substitutions)

    # Update the source if changes were made
    if processed_content != original_content:
        source[0] = processed_content
        logger.debug(f"Processed MyST substitutions in code blocks for {docname}")


def process_codeblock_substitutions(content: str, substitutions: dict) -> str:
    """
    Process MyST substitutions inside code blocks.

    This finds code blocks (```...```) and replaces {{ variable }} patterns
    with their values from myst_substitutions, but skips languages that
    commonly use {{ }} syntax natively.

    Uses a line-by-line approach to avoid regex backtracking issues.
    """
    # Languages that commonly use {{ }} syntax and should be skipped
    TEMPLATE_LANGUAGES = {
        "yaml",
        "yml",
        "helm",
        "jinja",
        "jinja2",
        "ansible",
        "j2",
        "go-template",
        "gotmpl",
        "handlebars",
        "hbs",
        "mustache",
        "django",
        "twig",
        "liquid",
        "smarty",
        "docker-compose",
    }

    lines = content.split("\n")
    result_lines = []
    in_code_block = False
    current_language = None
    code_block_lines = []

    for line in lines:
        if line.startswith("```") and not in_code_block:
            # Starting a code block
            language_match = re.match(r"```([a-zA-Z][a-zA-Z0-9_-]*)", line)
            if language_match:
                in_code_block = True
                current_language = language_match.group(1).lower()
                code_block_lines = [line]
            else:
                # Not a standard code block (might be a directive)
                result_lines.append(line)
        elif line == "```" and in_code_block:
            # Ending a code block
            code_block_lines.append(line)

            # Process the code block content
            if len(code_block_lines) > 2:  # Has content between start and end
                code_content = "\n".join(
                    code_block_lines[1:-1]
                )  # Content without fences

                # Skip template languages or template-like content
                if (
                    current_language not in TEMPLATE_LANGUAGES
                    and not is_likely_template_syntax(code_content)
                ):
                    # Replace substitutions in the code content
                    processed_code = replace_substitutions(code_content, substitutions)
                    result_lines.append(code_block_lines[0])  # Opening fence
                    result_lines.extend(processed_code.split("\n"))  # Processed content
                    result_lines.append(line)  # Closing fence
                else:
                    # For template languages, be more careful or skip
                    if current_language in TEMPLATE_LANGUAGES:
                        processed_code = replace_substitutions_carefully(
                            code_content, substitutions
                        )
                        result_lines.append(code_block_lines[0])  # Opening fence
                        result_lines.extend(
                            processed_code.split("\n")
                        )  # Processed content
                        result_lines.append(line)  # Closing fence
                    else:
                        # Add unchanged
                        result_lines.extend(code_block_lines)
            else:
                # Empty code block, add unchanged
                result_lines.extend(code_block_lines)

            # Reset state
            in_code_block = False
            current_language = None
            code_block_lines = []
        elif in_code_block:
            # Inside a code block, collect lines
            code_block_lines.append(line)
        else:
            # Regular content, add as-is
            result_lines.append(line)

    # Handle case where file ends while in a code block (malformed)
    if in_code_block and code_block_lines:
        result_lines.extend(code_block_lines)

    return "\n".join(result_lines)


def is_likely_template_syntax(content: str) -> bool:
    """
    Check if content looks like it contains template syntax that we shouldn't modify.

    Common patterns:
    - {{ .Values.something }} (Helm)
    - {{ ansible_variable }} (Ansible)
    - {{ item.property }} (loops)
    - {{- .Values.something }} (Helm with whitespace control)
    """
    template_patterns = [
        r"\{\{\s*\.[\w.]+\s*\}\}",  # {{ .Values.something }}
        r"\{\{\s*ansible_\w+\s*\}\}",  # {{ ansible_variable }}
        r"\{\{\s*item\.[\w.]+\s*\}\}",  # {{ item.property }}
        r"\{\{[-+]\s*[\w.]+\s*[-+]?\}\}",  # {{- variable }} or {{ variable -}}
        r"\{\{\s*\w+\.\w+",  # {{ object.property (general)
        r"\{\{\s*range\s+",  # {{ range ... }} (Go templates)
        r"\{\{\s*if\s+",  # {{ if ... }} (conditionals)
        r"\{\{\s*with\s+",  # {{ with ... }} (Go templates)
    ]

    for pattern in template_patterns:
        if re.search(pattern, content):
            return True

    return False


def replace_substitutions(text: str, substitutions: dict) -> str:
    """
    Replace {{ variable }} patterns with their values.
    """

    def replace_var(match):
        var_name = match.group(1).strip()
        if var_name in substitutions:
            replacement = str(substitutions[var_name])
            logger.debug(
                f"Replacing {{ {var_name} }} with '{replacement}' in code block"
            )
            return replacement
        else:
            logger.debug(f"Unknown substitution variable: {var_name}")
            return match.group(0)  # Return original if not found

    # Pattern to match {{ variable_name }} - only alphanumeric and underscore
    substitution_pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
    return re.sub(substitution_pattern, replace_var, text)


def replace_substitutions_carefully(text: str, substitutions: dict) -> str:
    """
    Replace {{ variable }} patterns with their values, but only for exact MyST variable matches.
    This is used for template languages where we want to avoid breaking existing template syntax.
    """

    def replace_var(match):
        full_match = match.group(0)
        var_name = match.group(1).strip()

        # Only replace if it's an exact match for one of our MyST variables
        if var_name in substitutions:
            # Double-check this isn't template syntax by looking for template patterns
            if not re.search(
                r"[.|\-+]", full_match
            ):  # No dots, pipes, or whitespace control
                replacement = str(substitutions[var_name])
                logger.debug(
                    f"Carefully replacing {{ {var_name} }} with '{replacement}' in template language"
                )
                return replacement

        # Leave everything else untouched
        return full_match

    # Pattern to match {{ variable_name }} - only alphanumeric and underscore
    substitution_pattern = r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}"
    return re.sub(substitution_pattern, replace_var, text)


def setup(app: Sphinx):
    """
    Setup function for the MyST code block substitution extension.
    """
    # Connect to the source-read event to process files before parsing
    app.connect("source-read", process_myst_source)

    return {
        "version": "1.0",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
