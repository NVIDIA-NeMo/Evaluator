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
"""nel package -- containerize a BYOB benchmark module."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import click

_DOCKERFILE_TEMPLATE = """\
FROM {base_image}

WORKDIR /app

RUN pip install --no-cache-dir nemo-evaluator

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt 2>/dev/null || true

COPY {module_name} /app/{module_name}

ENV PYTHONPATH=/app
ENTRYPOINT ["nel"]
"""


@click.command("package")
@click.option("--module", "-m", required=True, help="Python module path containing @benchmark + @scorer")
@click.option("--tag", "-t", required=True, help="Container image tag (e.g. my-bench:latest)")
@click.option("--base-image", default="python:3.11-slim", help="Base Docker image")
@click.option("--push", is_flag=True, help="Push image after build")
@click.option("--requirements", type=click.Path(exists=True), help="Extra requirements.txt")
@click.option("--output", "-o", type=click.Path(), help="Write Dockerfile to path instead of building")
def package_cmd(module, tag, base_image, push, requirements, output):
    """Package a BYOB benchmark module as a container image."""
    module_path = Path(module)
    if not module_path.exists():
        raise click.ClickException(f"Module path not found: {module}")

    module_name = module_path.name

    dockerfile_content = _DOCKERFILE_TEMPLATE.format(
        base_image=base_image,
        module_name=module_name,
    )

    if output:
        Path(output).write_text(dockerfile_content, encoding="utf-8")
        click.echo(f"Dockerfile written to {output}")
        return

    with tempfile.TemporaryDirectory(prefix="nel_package_") as tmpdir:
        dockerfile_path = Path(tmpdir) / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content, encoding="utf-8")

        req_path = Path(tmpdir) / "requirements.txt"
        if requirements:
            req_path.write_text(Path(requirements).read_text())
        else:
            req_path.write_text("")

        import shutil

        dest = Path(tmpdir) / module_name
        if module_path.is_dir():
            shutil.copytree(module_path, dest)
        else:
            shutil.copy2(module_path, dest)

        click.echo(f"Building image: {tag}")
        result = subprocess.run(
            ["docker", "build", "-t", tag, "-f", str(dockerfile_path), str(tmpdir)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise click.ClickException(f"Docker build failed:\n{result.stderr}")
        click.echo(f"Image built: {tag}")

        if push:
            click.echo(f"Pushing: {tag}")
            result = subprocess.run(["docker", "push", tag], capture_output=True, text=True)
            if result.returncode != 0:
                raise click.ClickException(f"Docker push failed:\n{result.stderr}")
            click.echo(f"Pushed: {tag}")
