# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Offline preview contract tests: python3 -m unittest discover -s .github/scripts."""

import copy
import json
import tempfile
import unittest
from pathlib import Path

from fern_preview import FERN_VERSION, preview_url, stage_content, verify_mirror


class PreviewTests(unittest.TestCase):
    """Exercise preview provenance and content-only staging."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / "incoming"
        self.trusted = self.root / "trusted"
        self.destination = self.root / "preview"
        for root in (self.artifact, self.trusted):
            (root / "docs/fern").mkdir(parents=True)
        (self.trusted / "docs/fern/docs.yml").write_text("instances: []\n")
        (self.trusted / "docs/index.mdx").write_text("old content")
        self.repo = "NVIDIA-NeMo/Evaluator"
        self.ref = "refs/heads/pull-request/42"
        self.sha = "a" * 40
        self.pr = {
            "number": 42,
            "state": "open",
            "head": {"repo": {"full_name": self.repo}, "sha": "a" * 40},
            "base": {"repo": {"full_name": self.repo}},
        }

    def test_normal_content_uses_trusted_configuration(self):
        (self.artifact / "docs/index.mdx").write_text("new content")
        (self.artifact / "docs/fern/fern.config.json").write_text(
            json.dumps({"version": "file:./fixture-cli", "organization": "fixture"})
        )
        (self.artifact / "docs/fern/package.json").write_text(
            json.dumps({"scripts": {"generate:library": "echo fixture", "sanitize:generated": "echo fixture"}})
        )
        (self.artifact / "docs/fern/docs.yml").write_text("instances: [fixture]\n")
        (self.artifact / "docs/fern/.npmrc").write_text("registry=https://example.invalid\n")
        (self.artifact / "docs/fern/tool.js").write_text("throw new Error('fixture');")
        stage_content(self.artifact, self.trusted, self.destination)
        docs = self.destination / "docs"
        self.assertEqual((docs / "index.mdx").read_text(), "new content")
        self.assertEqual((docs / "fern/docs.yml").read_text(), "instances: []\n")
        self.assertEqual(
            json.loads((docs / "fern/fern.config.json").read_text()),
            {
                "version": FERN_VERSION,
                "organization": "nvidia",
            },
        )
        for name in ("package.json", ".npmrc", "tool.js"):
            self.assertFalse((docs / "fern" / name).exists())

    def test_publisher_uses_fixed_cli_and_mirror(self):
        workflow = (Path(__file__).parents[1] / "workflows/fern-docs-preview.yml").read_text()
        for value in (
            f"fern-api@{FERN_VERSION}",
            "--ignore-scripts",
            "ref: main",
            "persist-credentials: false",
            "pull-request/[0-9]+",
            "docs/**",
            ".github/workflows/fern-docs-ci.yml",
            "same_repository",
        ):
            self.assertIn(value, workflow)
        for value in (
            "npm run",
            "npx",
            "jq -r .version",
            "get-slug-for-file",
            "workflow_run:",
            "download-artifact",
            "upload-artifact",
        ):
            self.assertNotIn(value, workflow)
        self.assertEqual(workflow.count("FERN_TOKEN:"), 1)

    def test_deleted_pages_are_not_restored(self):
        stage_content(self.artifact, self.trusted, self.destination)
        self.assertFalse((self.destination / "docs/index.mdx").exists())

    def test_links_are_rejected(self):
        (self.artifact / "docs/page.mdx").symlink_to(self.trusted / "docs/index.mdx")
        with self.assertRaises(ValueError):
            stage_content(self.artifact, self.trusted, self.destination)

    def test_current_mirror(self):
        self.assertEqual(verify_mirror(self.ref, self.sha, self.pr, self.repo), 42)

    def test_invalid_mirror_refs_and_sha(self):
        for ref in (
            "refs/heads/main",
            "refs/heads/pull-request/0",
            "refs/heads/pull-request/43",
            "refs/heads/pull-request/42/suffix",
            "refs/heads/pull-request/042",
        ):
            with self.subTest(ref=ref), self.assertRaises(ValueError):
                verify_mirror(ref, self.sha, self.pr, self.repo)
        with self.assertRaises(ValueError):
            verify_mirror(self.ref, "invalid", self.pr, self.repo)

    def test_forks_stale_closed_and_wrong_base(self):
        for mutation in ("fork", "stale", "closed", "base", "number"):
            pr = copy.deepcopy(self.pr)
            if mutation == "fork":
                pr["head"]["repo"]["full_name"] = "example/Evaluator"
            elif mutation == "stale":
                pr["head"]["sha"] = "b" * 40
            elif mutation == "closed":
                pr["state"] = "closed"
            elif mutation == "base":
                pr["base"]["repo"]["full_name"] = "example/Evaluator"
            else:
                pr["number"] = "42"
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                verify_mirror(self.ref, self.sha, pr, self.repo)

    def test_url_is_display_only_and_requires_fern_https(self):
        url = "https://nvidia-preview-example.docs.buildwithfern.com/nemo/evaluator"
        self.assertEqual(preview_url(f"Published docs to {url} (preview)"), url)
        for value in (
            "https://example.invalid",
            "https://a.docs.buildwithfern.com.evil.invalid",
            "https://user@a.docs.buildwithfern.com",
            "http://a.docs.buildwithfern.com",
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                preview_url(f"Published docs to {value} (preview)")


if __name__ == "__main__":
    unittest.main()
