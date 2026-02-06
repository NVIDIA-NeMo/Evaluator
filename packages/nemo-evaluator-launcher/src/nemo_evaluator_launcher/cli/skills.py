# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from simple_parsing import field

from nemo_evaluator_launcher.common.logging_utils import logger
from nemo_evaluator_launcher.package_info import __version__

_REPO = "NVIDIA-NeMo/Evaluator"
_SKILLS_REPO_PATH = "packages/nemo-evaluator-launcher/.claude/skills"

_AGENT_PATHS: dict[str, tuple[Path, Path]] = {
    # (user-level, project-level)
    "claude": (Path.home() / ".claude" / "skills", Path(".claude") / "skills"),
    "codex": (Path.home() / ".agents" / "skills", Path(".agents") / "skills"),
    "cursor": (Path.home() / ".cursor" / "skills", Path(".cursor") / "skills"),
    "opencode": (
        Path.home() / ".config" / "opencode" / "skills",
        Path(".opencode") / "skills",
    ),
}

_ALL_AGENTS = list(_AGENT_PATHS)

_SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _progress(skill_name: str, step: int, total: int) -> None:
    """Print an inline progress indicator with spinner, or a final checkmark when *done*."""
    pct = step * 100 // total if total else 100
    if step == total:
        print(f"\r  ✓ Downloaded {skill_name} {step}/{total} ({pct}%)          ")
    else:
        frame = _SPINNER[step % len(_SPINNER)]
        print(
            f"\r  {frame} Downloading {skill_name} {step}/{total} ({pct}%)",
            end="",
            flush=True,
        )


def _github_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "nemo-evaluator-launcher",
    }
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


_RATE_LIMIT_HINT = (
    "GitHub API rate limit exceeded. Set a personal access token to increase the limit:\n"
    "  export GITHUB_TOKEN=ghp_...\n"
    "Generate one at https://github.com/settings/tokens"
)


def _github_list_dir(path: str, ref: str) -> list[dict]:
    url = f"https://api.github.com/repos/{_REPO}/contents/{path}?ref={ref}"
    resp = requests.get(url, timeout=30, headers=_github_headers())
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise RuntimeError(_RATE_LIMIT_HINT)
    if resp.status_code != 200:
        raise RuntimeError(f"GitHub API error ({resp.status_code}): {resp.text}")
    data = resp.json()
    return [data] if isinstance(data, dict) else data


def _collect_files(path: str, ref: str) -> list[dict]:
    """Recursively list all files under *path* via GitHub Contents API."""
    files: list[dict] = []
    for item in _github_list_dir(path, ref):
        if item["type"] == "dir":
            files.extend(_collect_files(item["path"], ref))
        else:
            files.append(item)
    return files


def _discover_skills(ref: str) -> list[str]:
    """Return skill directory names found under the skills repo path."""
    items = _github_list_dir(_SKILLS_REPO_PATH, ref)
    return [item["name"] for item in items if item["type"] == "dir"]


def _download_skill(skill_name: str, ref: str, dest: Path) -> None:
    """Download a single skill tree from GitHub into *dest*."""
    skill_repo_path = f"{_SKILLS_REPO_PATH}/{skill_name}"
    files = _collect_files(skill_repo_path, ref)
    total = len(files)

    for i, item in enumerate(files):
        rel = Path(item["path"]).relative_to(skill_repo_path)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        dl_url = item.get("download_url")
        if not dl_url:
            raise RuntimeError(f"No download_url for {item['path']}")
        logger.debug("Downloading skill file", file=str(rel), url=dl_url)
        r = requests.get(dl_url, timeout=30, headers=_github_headers())
        if r.status_code != 200:
            raise RuntimeError(f"Download failed ({r.status_code}): {dl_url}")
        target.write_bytes(r.content)
        _progress(skill_name, i + 1, total)


@dataclass
class InstallCmd:
    """Install NEL agent skills for AI coding assistants.

    Examples:
      nel skills install --claude
      nel skills install --claude --codex --cursor --opencode
      nel skills install --cursor --project
      nel skills install --ref main --claude
    """

    claude: bool = field(
        default=False, action="store_true", help="Install for Claude Code"
    )
    codex: bool = field(default=False, action="store_true", help="Install for Codex")
    cursor: bool = field(default=False, action="store_true", help="Install for Cursor")
    opencode: bool = field(
        default=False, action="store_true", help="Install for OpenCode"
    )
    project: bool = field(
        default=False,
        action="store_true",
        help="Install into project directory instead of user home",
    )
    ref: Optional[str] = field(
        default=None,
        help="Git ref (branch, tag, sha). Defaults to the tag matching the installed nel version.",
    )
    force: bool = field(
        default=False,
        action="store_true",
        help="Overwrite existing skill directories",
    )

    def execute(self) -> None:
        ref = self.ref or f"nemo-evaluator-launcher-v{__version__}"
        agents = [a for a in _ALL_AGENTS if getattr(self, a)] or _ALL_AGENTS

        print(f"Fetching skill list from {_REPO} (ref: {ref}) ...")
        skill_names = _discover_skills(ref)
        if not skill_names:
            raise RuntimeError(f"No skills found at {_SKILLS_REPO_PATH} (ref: {ref})")

        for skill_name in skill_names:
            for agent in agents:
                user_dir, project_dir = _AGENT_PATHS[agent]
                root = project_dir if self.project else user_dir
                dest = root / skill_name

                if dest.exists() and not self.force:
                    print(f"Already installed: {dest} (use --force to overwrite)")
                    continue

                if dest.exists():
                    shutil.rmtree(dest)

                dest.mkdir(parents=True, exist_ok=True)
                _download_skill(skill_name, ref, dest)
                print(f"Installed {skill_name} -> {dest} (ref: {ref})")
