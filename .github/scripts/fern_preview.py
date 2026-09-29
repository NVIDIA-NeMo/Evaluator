# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage mirror docs without importing PR tooling or configuration."""

import argparse
import json
import os
import re
import shutil
from pathlib import Path

FERN_VERSION = "5.29.0"
CONTENT_ROOTS = ("docs",)
FERN_DIRECTORY = "docs/fern"
CONTENT_SUFFIXES = {".md", ".mdx", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico", ".pdf"}


def verify_mirror(ref: str, sha: str, pr: dict, repository: str) -> int:
    """Bind an approved mirror push to a current, same-repository open PR."""
    match = re.fullmatch(r"refs/heads/pull-request/([1-9][0-9]*)", ref)
    number = pr["number"]
    if (
        not match
        or type(number) is not int
        or number != int(match[1])
        or not re.fullmatch(r"[0-9a-f]{40}", sha)
        or pr["state"] != "open"
        or pr["head"]["sha"] != sha
        or pr["head"]["repo"]["full_name"] != repository
        or pr["base"]["repo"]["full_name"] != repository
    ):
        raise ValueError("Mirror does not identify a current same-repository PR")
    return number


def stage_content(artifact: Path, trusted: Path, destination: Path) -> None:
    """Overlay document content onto trusted docs; never copy executable tooling."""
    # Reject links even when their suffix would otherwise be ignored.
    for root in CONTENT_ROOTS:
        source = artifact / root
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError("Content links are not supported")
        if not source.is_dir():
            raise ValueError(f"Missing content directory: {root}")
        shutil.copytree(trusted / root, destination / root, symlinks=False)
        # Removed PR pages must not silently reappear from the trusted checkout.
        for path in (destination / root).rglob("*"):
            if path.is_file() and path.suffix.lower() in CONTENT_SUFFIXES:
                path.unlink()
        for path in source.rglob("*"):
            relative = path.relative_to(source)
            if any(part.startswith(".") for part in relative.parts):
                continue
            if path.is_file() and path.suffix.lower() in CONTENT_SUFFIXES:
                target = destination / root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(path.read_bytes())
    config = destination / FERN_DIRECTORY / "fern.config.json"
    config.write_text(json.dumps({"organization": "nvidia", "version": FERN_VERSION}) + "\n")


def preview_url(output: str) -> str:
    """Accept one HTTPS Fern URL for display only; never send it credentials."""
    urls = re.findall(r"Published docs to (https://[^\s<>]+) \(", output)
    if len(urls) != 1:
        raise ValueError("Expected one preview URL")
    from urllib.parse import urlsplit

    parsed = urlsplit(urls[0])
    if (
        parsed.username
        or parsed.password
        or parsed.port is not None
        or not parsed.hostname
        or not re.fullmatch(r"nvidia-preview-[a-z0-9-]+\.docs\.buildwithfern\.com", parsed.hostname)
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("Unexpected preview URL")
    return urls[0]


def main() -> None:
    """Validate preview inputs or emit a display-only URL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["verify", "stage", "url"])
    parser.add_argument("--artifact", type=Path)
    parser.add_argument("--trusted", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--ref")
    parser.add_argument("--sha")
    parser.add_argument("--pr", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.command == "url":
        outputs = {"preview_url": preview_url(args.log.read_text())}
    else:
        pr = json.loads(args.pr.read_text())
        number = verify_mirror(args.ref, args.sha, pr, os.environ["GITHUB_REPOSITORY"])
        if args.command == "stage":
            stage_content(args.artifact, args.trusted, args.destination)
        outputs = {"pr_number": str(number), "preview_id": f"pr-{number}-{args.sha[:12]}"}
    with open(os.environ["GITHUB_OUTPUT"], "a") as stream:
        for key, value in outputs.items():
            stream.write(f"{key}={value}\n")


if __name__ == "__main__":
    main()
