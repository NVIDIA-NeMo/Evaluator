# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""nel cache-sqsh — build a .sqsh image on a SLURM cluster."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import click

_KNOWN_TARGETS = {
    "base": "docker/Dockerfile.base",
    "harbor": "docker/Dockerfile.harbor",
    "lm-eval": "docker/Dockerfile.lm-eval",
    "skills": "docker/Dockerfile.skills",
    "gym": "docker/Dockerfile.gym",
    "full": "docker/Dockerfile.full",
}

# TODO: verify this default registry points to the correct public registry
_DEFAULT_REGISTRY = "nvcr.io/nvidia/nemo-evaluator"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    click.echo(f"  $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, text=True, **kwargs)


@click.command("cache-sqsh")
@click.argument("target")
@click.argument("host")
@click.argument("output_path")
@click.option(
    "--registry",
    "-r",
    default=_DEFAULT_REGISTRY,
    help="Docker registry to push NEL images to.",
)
@click.option(
    "--tag",
    "-t",
    default="local",
    help="Tag for the pushed image (default: local).",
)
@click.option(
    "--filename",
    "-f",
    default=None,
    help="Output .sqsh filename (default: derived from target).",
)
def cache_sqsh_cmd(
    target: str,
    host: str,
    output_path: str,
    registry: str,
    tag: str,
    filename: str | None,
) -> None:
    """Build/pull a Docker image and create a .sqsh on a SLURM cluster.

    TARGET is either a built-in NEL variant (base, harbor, lm-eval, skills,
    gym, full) which builds from local source and pushes to the registry, or
    a Docker image URI (e.g. vllm/vllm-openai:v0.18.0-cu130).

    HOST is the SLURM login node hostname.

    OUTPUT_PATH is the remote directory to store the .sqsh file (on Lustre).

    \b
    Examples:
      # Build harbor from local source, push to registry, sqsh on cluster:
      nel cache-sqsh harbor cw-dfw-cs-001-login-01 \\
          /lustre/fsw/portfolios/coreai/users/\\$USER/cache/nel

      # Create sqsh of a third-party image on cluster:
      nel cache-sqsh vllm/vllm-openai:v0.18.0-cu130 cw-dfw-cs-001-login-01 \\
          /lustre/fsw/portfolios/coreai/users/\\$USER/cache/vllm
    """
    if shutil.which("docker") is None:
        click.secho("Error: 'docker' not found.", fg="red", err=True)
        sys.exit(1)

    is_local_build = target in _KNOWN_TARGETS

    if is_local_build:
        dockerfile = _KNOWN_TARGETS[target]
        project_root = _find_project_root()
        if not (project_root / dockerfile).exists():
            click.secho(f"Error: {dockerfile} not found in {project_root}", fg="red", err=True)
            sys.exit(1)
        remote_image = f"{registry}:{tag}-{target}"
        if filename is None:
            filename = f"nel-{target}.sqsh"
    else:
        remote_image = target
        if filename is None:
            filename = target.replace("/", "-").replace(":", "-") + ".sqsh"

    steps = 3 if is_local_build else 1
    step = 0

    if is_local_build:
        step += 1
        click.secho(f"\n[{step}/{steps}] Building {target} from {dockerfile} ...", bold=True)
        _run(
            [
                "docker",
                "build",
                "-f",
                str(project_root / dockerfile),
                "-t",
                remote_image,
                str(project_root),
            ]
        )

        step += 1
        click.secho(f"[{step}/{steps}] Pushing to {remote_image} ...", bold=True)
        _run(["docker", "push", remote_image])

    step += 1
    enroot_uri = f"docker://{remote_image}" if "://" not in remote_image else remote_image
    sqsh_path = f"{output_path}/{filename}"
    click.secho(f"[{step}/{steps}] Creating sqsh on {host} ...", bold=True)
    _run(
        [
            "ssh",
            host,
            f"mkdir -p '{output_path}' && enroot import -o '{sqsh_path}' -- '{enroot_uri}'",
        ]
    )

    size = subprocess.run(
        ["ssh", host, f"du -h '{sqsh_path}' | cut -f1"],
        capture_output=True,
        text=True,
    )
    if size.returncode == 0:
        click.echo(f"  sqsh size: {size.stdout.strip()}")

    click.secho(f"\nDone. Use in config:\n  image: {sqsh_path}\n", fg="green")


def _find_project_root() -> Path:
    """Walk up from CWD to find pyproject.toml."""
    p = Path.cwd()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "docker").is_dir():
            return parent
    click.secho("Error: cannot find project root (pyproject.toml + docker/)", fg="red", err=True)
    sys.exit(1)
