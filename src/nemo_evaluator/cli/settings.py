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
"""nel config -- persistent user configuration."""

from __future__ import annotations

from pathlib import Path

import click
import yaml

_CONFIG_DIR = Path.home() / ".config" / "nemo-evaluator"
_CONFIG_FILE = _CONFIG_DIR / "config.yaml"


def _load() -> dict:
    if _CONFIG_FILE.exists():
        return yaml.safe_load(_CONFIG_FILE.read_text()) or {}
    return {}


def _save(data: dict) -> None:
    _CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    _CONFIG_FILE.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")


def _get_nested(data: dict, key: str):
    parts = key.split(".")
    for p in parts:
        if not isinstance(data, dict) or p not in data:
            return None
        data = data[p]
    return data


def _set_nested(data: dict, key: str, value: str) -> None:
    parts = key.split(".")
    d = data
    for p in parts[:-1]:
        d = d.setdefault(p, {})
    d[parts[-1]] = value


@click.group("config")
def config_cmd():
    """Manage persistent configuration."""


@config_cmd.command("show")
def config_show():
    """Display all configuration."""
    data = _load()
    if not data:
        click.echo("No configuration set.")
        click.echo(f"Config file: {_CONFIG_FILE}")
        return
    click.echo(yaml.dump(data, default_flow_style=False).rstrip())
    click.echo(f"\nConfig file: {_CONFIG_FILE}")


@config_cmd.command("get")
@click.argument("key")
def config_get(key):
    """Get a configuration value."""
    data = _load()
    value = _get_nested(data, key)
    if value is None:
        raise click.ClickException(f"Key not found: {key}")
    click.echo(value)


@config_cmd.command("set")
@click.argument("key")
@click.argument("value")
def config_set(key, value):
    """Set a configuration value."""
    data = _load()
    _set_nested(data, key, value)
    _save(data)
    click.echo(f"{key} = {value}")


@config_cmd.command("unset")
@click.argument("key")
def config_unset(key):
    """Remove a configuration value."""
    data = _load()
    parts = key.split(".")
    d = data
    for p in parts[:-1]:
        if p not in d:
            raise click.ClickException(f"Key not found: {key}")
        d = d[p]
    if parts[-1] not in d:
        raise click.ClickException(f"Key not found: {key}")
    del d[parts[-1]]
    _save(data)
    click.echo(f"Removed: {key}")


def get_persistent_defaults() -> dict:
    """Load persistent config defaults for use in CLI and EvalConfig."""
    return _load()
