# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for API reference MDX generation helpers."""

from __future__ import annotations

import unittest

from generate_api_reference_mdx import _clean_docstring, _clear_caches, _slugify


class CleanDocstringTests(unittest.TestCase):
    def test_args_returns_bullets(self) -> None:
        text = """Summary line.

Args:
    endpoint_url (str): Full URL with ``/v1/chat/completions``
    model_name (str): Model identifier

Returns:
    bool: Whether the endpoint is alive
"""
        result = _clean_docstring(text)
        self.assertIn("**Args**", result)
        self.assertIn("- endpoint_url (str):", result)
        self.assertIn("`/v1/chat/completions`", result)
        self.assertIn("**Returns**", result)
        self.assertIn("- bool:", result)

    def test_rst_literal_block(self) -> None:
        text = """Prints tasks in the format of::

    {harness1}:
        * benchmark A
"""
        result = _clean_docstring(text)
        self.assertIn("format of:", result)
        self.assertIn("```", result)
        self.assertIn("{harness1}:", result)
        self.assertNotIn("format of::", result)

    def test_important_admonition(self) -> None:
        text = ".. important:: Only installed wheels are returned."
        result = _clean_docstring(text)
        self.assertIn("> **Important:**", result)

    def test_inline_important(self) -> None:
        text = "Required interceptor.\nImportant: Keep this last."
        result = _clean_docstring(text)
        self.assertIn("> **Important:** Keep this last.", result)


class SlugifyTests(unittest.TestCase):
    def test_class_anchor(self) -> None:
        self.assertEqual(_slugify("CachingInterceptor"), "cachinginterceptor")


class CacheTests(unittest.TestCase):
    def test_clear_caches_is_idempotent(self) -> None:
        _clear_caches()
        _clear_caches()


if __name__ == "__main__":
    unittest.main()
