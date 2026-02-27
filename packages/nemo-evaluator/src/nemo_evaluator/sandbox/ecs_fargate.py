# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""ECS Fargate sandbox.

Supports two transport modes over SSH:

  - **Exec-server mode** — one-way SSH tunnel.  An embedded HTTP exec
    server runs inside the container; the orchestrator drives all
    command execution, uploads, and downloads through it.

  - **Agent-server mode** — two-way SSH tunnel.  The container hosts a
    self-contained agent that reaches the model via a reverse tunnel;
    the orchestrator connects to the agent's API via a forward tunnel.

Includes Docker image building via AWS CodeBuild with ECR caching.
"""

from __future__ import annotations

import atexit
import base64
import io
import json
import logging
import os
import random
import re
import shlex
import socket
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, TypeVar
from urllib.parse import urlparse

from .base import ExecResult

log = logging.getLogger(__name__)
T = TypeVar("T")


# =====================================================================
# Lazy AWS SDK import
# =====================================================================


def _require_aws_sdks():
    """Import boto3/botocore only when actually needed."""
    try:
        import importlib

        boto3 = importlib.import_module("boto3")
        botocore_config = importlib.import_module("botocore.config")
        botocore_exceptions = importlib.import_module("botocore.exceptions")
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "ECS Fargate sandbox requires boto3/botocore. "
            "Install them (`pip install boto3`) or use a different sandbox backend."
        ) from e

    Config = getattr(botocore_config, "Config")
    ClientError = getattr(botocore_exceptions, "ClientError")
    return boto3, Config, ClientError


# =====================================================================
# Config dataclasses
# =====================================================================


def _coerce_list(value: Any, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, str):
        return [value]
    raise ValueError(f"{name} must be a list, got {type(value)!r}")


def _sanitize_id(value: str, max_len: int = 100) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9-]+", "-", value).strip("-")
    return cleaned[:max_len] or "task"


@dataclass(frozen=True)
class SshSidecarConfig:
    """SSH sidecar container configuration.

    The sidecar runs sshd in an Alpine container alongside the main
    container, providing SSH-based transport for tunnelling and execution.

    Two modes depending on ``exec_server_port``:

    * **Exec-server mode** (``exec_server_port`` is set):
      One-way SSH tunnel.  The sandbox uploads a zero-dependency HTTP
      exec server into the main container and forwards a local port
      to it.  ``exec()``, ``upload()``, ``download()`` all work.

    * **Agent-server mode** (``exec_server_port`` is ``None``):
      Two-way SSH tunnel.  A reverse tunnel (``-R``) makes the model
      endpoint reachable inside the task; a forward tunnel (``-L``)
      gives the orchestrator access to the agent server.  The consumer
      is responsible for command execution via its own agent API.
    """

    sshd_port: int = 2222
    ssh_ready_timeout_sec: float = 120.0
    public_key_secret_arn: str = ""  # required — pre-provisioned only
    private_key_secret_arn: str = ""  # required — pre-provisioned only
    image: str | None = None  # sidecar image (None → alpine:latest)

    # Two-way tunnel config (agent-server mode)
    target_url_env: str = "MODEL_URL,MODEL_BASE_URL"
    local_port: int = 0  # port inside task for model; 0 = infer from target URL
    model_env_var: str = "MODEL_BASE_URL"

    # Exec server config (exec-server mode; None → agent-server mode)
    exec_server_port: int | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> SshSidecarConfig:
        return cls(
            sshd_port=int(raw.get("sshd_port", 2222)),
            ssh_ready_timeout_sec=float(raw.get("ssh_ready_timeout_sec", 120.0)),
            public_key_secret_arn=str(raw.get("public_key_secret_arn", "")),
            private_key_secret_arn=str(raw.get("private_key_secret_arn", "")),
            image=raw.get("image"),
            target_url_env=str(raw.get("target_url_env", "MODEL_URL,MODEL_BASE_URL")),
            local_port=int(raw.get("local_port", 0)),
            model_env_var=str(raw.get("model_env_var", "MODEL_BASE_URL")),
            exec_server_port=(
                int(raw["exec_server_port"])
                if raw.get("exec_server_port") is not None
                else None
            ),
        )


@dataclass(frozen=True)
class EnvVarSpec:
    """Name–value pair for explicit environment variable injection."""

    name: str
    value: str

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EnvVarSpec:
        name = str(raw.get("name") or "")
        value = str(raw.get("value") or "")
        if not name:
            raise ValueError("EnvVarSpec requires 'name'")
        return cls(name=name, value=value)


@dataclass(frozen=True)
class EcsFargateConfig:
    """Configuration for the ECS Fargate sandbox."""

    # AWS infrastructure
    region: str | None = None
    cluster: str = ""
    subnets: list[str] = field(default_factory=list)
    security_groups: list[str] = field(default_factory=list)
    assign_public_ip: bool = False

    # Task definition
    task_definition: str | None = None  # base task def to clone
    task_definition_family_prefix: str = "ecs-sandbox"
    image_template: str | None = None  # supports {task_id}, {task_id_sanitized}
    container_name: str = "main"
    container_port: int | None = None  # agent-server port (agent-server mode)
    cpu: str = "4096"
    memory: str = "8192"
    ephemeral_storage_gib: int | None = None
    platform_version: str | None = None
    execution_role_arn: str | None = None
    task_role_arn: str | None = None
    extra_env: dict[str, str] | None = None

    # Logging
    log_group: str | None = None
    log_stream_prefix: str | None = None

    # Lifecycle
    max_task_lifetime_sec: int = 14400  # 4 h
    startup_timeout_sec: float = 300.0
    poll_interval_sec: float = 2.0
    run_task_max_retries: int = 30

    # SSH sidecar
    ssh_sidecar: SshSidecarConfig | None = None

    # Model endpoint injection (agent-server mode)
    model_endpoint_env: EnvVarSpec | None = None

    # S3 file staging
    s3_bucket: str | None = None
    s3_prefix: str | None = None

    # Docker build via AWS CodeBuild
    ecr_repository: str | None = None
    environment_dir: str | None = None
    codebuild_project: str | None = None
    codebuild_service_role: str | None = None
    codebuild_compute_type: str = "BUILD_GENERAL1_MEDIUM"
    codebuild_build_timeout: int = 30
    dockerhub_secret_arn: str | None = None
    build_parallelism: int = 50

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> EcsFargateConfig:
        subnets = _coerce_list(raw.get("subnets"), "subnets")
        sgs = _coerce_list(raw.get("security_groups"), "security_groups")
        has_sidecar = isinstance(raw.get("ssh_sidecar"), Mapping)
        assign_public_ip = bool(raw.get("assign_public_ip", False)) or has_sidecar

        return cls(
            region=raw.get("region"),
            cluster=str(raw.get("cluster", "")),
            subnets=subnets,
            security_groups=sgs,
            assign_public_ip=assign_public_ip,
            task_definition=raw.get("task_definition"),
            task_definition_family_prefix=str(
                raw.get("task_definition_family_prefix", "ecs-sandbox")
            ),
            image_template=raw.get("image_template"),
            container_name=str(raw.get("container_name", "main")),
            container_port=(
                int(raw["container_port"])
                if raw.get("container_port") is not None
                else None
            ),
            cpu=str(raw.get("cpu", "4096")),
            memory=str(raw.get("memory", "8192")),
            ephemeral_storage_gib=(
                int(raw["ephemeral_storage_gib"])
                if raw.get("ephemeral_storage_gib") is not None
                else None
            ),
            platform_version=raw.get("platform_version"),
            execution_role_arn=raw.get("execution_role_arn"),
            task_role_arn=raw.get("task_role_arn"),
            extra_env=(
                {str(k): str(v) for k, v in raw["extra_env"].items()}
                if isinstance(raw.get("extra_env"), Mapping)
                else None
            ),
            log_group=raw.get("log_group"),
            log_stream_prefix=raw.get("log_stream_prefix"),
            max_task_lifetime_sec=int(raw.get("max_task_lifetime_sec", 14400)),
            startup_timeout_sec=float(raw.get("startup_timeout_sec", 300.0)),
            poll_interval_sec=float(raw.get("poll_interval_sec", 2.0)),
            run_task_max_retries=int(raw.get("run_task_max_retries", 30)),
            ssh_sidecar=(
                SshSidecarConfig.from_dict(raw["ssh_sidecar"]) if has_sidecar else None
            ),
            model_endpoint_env=(
                EnvVarSpec.from_dict(raw["model_endpoint_env"])
                if isinstance(raw.get("model_endpoint_env"), Mapping)
                else None
            ),
            s3_bucket=raw.get("s3_bucket"),
            s3_prefix=raw.get("s3_prefix"),
            ecr_repository=raw.get("ecr_repository"),
            environment_dir=raw.get("environment_dir"),
            codebuild_project=raw.get("codebuild_project"),
            codebuild_service_role=raw.get("codebuild_service_role"),
            codebuild_compute_type=str(
                raw.get("codebuild_compute_type", "BUILD_GENERAL1_MEDIUM")
            ),
            codebuild_build_timeout=int(raw.get("codebuild_build_timeout", 30)),
            dockerhub_secret_arn=raw.get("dockerhub_secret_arn"),
            build_parallelism=int(raw.get("build_parallelism", 50)),
        )


# =====================================================================
# Retry utilities
# =====================================================================

_RETRYABLE_CODES = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailable",
        "RequestLimitExceeded",
    }
)

_RETRYABLE_MESSAGES = (
    "Capacity is unavailable",
    "Rate exceeded",
    "Too many concurrent",
    "throttl",
    "connect timeout",
    "read timeout",
    "connection reset",
    "endpointconnectionerror",
)


def is_retryable_error(exc: Exception) -> bool:
    """Return *True* if *exc* looks like a transient AWS error."""
    msg = str(exc).lower()
    code = ""
    if hasattr(exc, "response"):
        code = (exc.response.get("Error") or {}).get("Code", "")  # type: ignore[union-attr]
    return code in _RETRYABLE_CODES or any(m in msg for m in _RETRYABLE_MESSAGES)


def retry_with_backoff(
    func: Callable[[], T],
    *,
    operation_name: str,
    max_retries: int = 0,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter: float = 0.5,
) -> T:
    """Call *func* with exponential back-off on retryable errors.

    *max_retries* = 0 means infinite retries.
    """
    attempt = 0
    while True:
        try:
            return func()
        except Exception as exc:
            if not is_retryable_error(exc):
                raise
            attempt += 1
            if 0 < max_retries <= attempt:
                log.error(
                    "%s failed after %d retries: %s", operation_name, attempt, exc
                )
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            delay *= 1 + random.uniform(-jitter, jitter)
            log.warning(
                "%s throttled (attempt %d), retrying in %.1fs: %s",
                operation_name,
                attempt,
                delay,
                exc,
            )
            time.sleep(delay)


# =====================================================================
# SSH helpers — secrets & port allocation
# =====================================================================


def _free_port() -> int:
    """Allocate an ephemeral TCP port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def download_secret_to_file(secret_arn: str, region: str | None = None) -> str:
    """Fetch a Secrets Manager secret and write it to a temp file (mode 0600).

    The caller is responsible for deleting the file.
    """
    boto3, *_ = _require_aws_sdks()
    sm = boto3.client("secretsmanager", region_name=region)
    resp = sm.get_secret_value(SecretId=secret_arn)
    key_material: str = resp["SecretString"]

    fd, path = tempfile.mkstemp(prefix="ecs-ssh-", suffix=".key")
    try:
        os.write(fd, key_material.encode())
    finally:
        os.close(fd)
    os.chmod(path, 0o600)
    log.debug("Downloaded SSH key to %s", path)
    return path


def download_secret_to_string(secret_arn: str, region: str | None = None) -> str:
    """Fetch a Secrets Manager secret and return it as a string."""
    boto3, *_ = _require_aws_sdks()
    sm = boto3.client("secretsmanager", region_name=region)
    resp = sm.get_secret_value(SecretId=secret_arn)
    return resp["SecretString"]


# =====================================================================
# SSH tunnel
# =====================================================================


class SshTunnel:
    """Manages an ``ssh -N`` subprocess with ``-L`` and/or ``-R`` tunnels.

    Two usage patterns:

    **Exec-server mode** — forward a single remote port::

        tunnel = SshTunnel(host=ip, port=2222, key_file=key,
                           forward_port=19542)
        tunnel.open()
        # tunnel.local_port → auto-allocated ephemeral port

    **Agent-server mode** — explicit forward + reverse specs::

        fwd = _free_port()
        tunnel = SshTunnel(host=ip, port=2222, key_file=key,
                           forwards=[f"{fwd}:localhost:8000"],
                           reverses=[f"11434:model-host:11434"])
        tunnel.open()
        # tunnel.local_port → fwd
    """

    def __init__(
        self,
        *,
        host: str,
        port: int = 2222,
        user: str = "root",
        key_file: str,
        forward_port: int | None = None,
        forwards: list[str] | None = None,
        reverses: list[str] | None = None,
        local_port_override: int | None = None,
    ) -> None:
        self._host = host
        self._port = port
        self._user = user
        self._key_file = key_file
        self._simple_forward_port = forward_port
        self._forwards = list(forwards or [])
        self._reverses = list(reverses or [])
        self._local_port: int | None = local_port_override
        self._proc: subprocess.Popen[bytes] | None = None

    @property
    def local_port(self) -> int:
        if self._local_port is None:
            raise RuntimeError("Tunnel not open yet — call open() first")
        return self._local_port

    @property
    def is_open(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def open(self, *, max_retries: int = 15, initial_backoff: float = 5.0) -> None:
        """Start the SSH tunnel with retries for transient connection errors."""
        if self.is_open:
            return

        # In simple mode, allocate a fresh local port per attempt.
        use_simple = self._simple_forward_port is not None

        last_err = ""
        backoff = initial_backoff
        for attempt in range(1, max_retries + 1):
            if use_simple:
                self._local_port = _free_port()

            cmd = self._build_ssh_cmd()
            log.info(
                "SSH tunnel attempt %d/%d: %s", attempt, max_retries, " ".join(cmd)
            )

            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            time.sleep(3)

            if self._proc.poll() is None:
                # Process alive — verify forward port is reachable.
                if self._local_port:
                    try:
                        self._wait_for_local_port(self._local_port, timeout=15.0)
                    except Exception as port_exc:
                        log.warning(
                            "SSH alive but forward port %d not open: %s",
                            self._local_port,
                            port_exc,
                        )
                        self._kill()
                        last_err = str(port_exc)
                        time.sleep(min(5.0, attempt * 1.5))
                        continue
                log.info(
                    "SSH tunnel started (pid=%d, attempt %d/%d)",
                    self._proc.pid,
                    attempt,
                    max_retries,
                )
                return

            stderr = (
                self._proc.stderr.read().decode(errors="replace")
                if self._proc.stderr
                else ""
            )
            last_err = stderr.strip()
            self._proc = None

            if not any(
                m in last_err
                for m in (
                    "Connection refused",
                    "Connection timed out",
                    "No route to host",
                    "Connection reset",
                )
            ):
                raise RuntimeError(
                    f"SSH tunnel exited immediately (attempt {attempt}): {last_err}"
                )

            log.warning(
                "SSH tunnel attempt %d/%d failed: %s — retrying in %.0fs",
                attempt,
                max_retries,
                last_err,
                backoff,
            )
            time.sleep(backoff)
            backoff = min(30.0, backoff * 1.5)

        raise RuntimeError(
            f"SSH tunnel failed after {max_retries} attempts: {last_err}"
        )

    def close(self) -> None:
        """Terminate the SSH tunnel subprocess."""
        self._kill()

    def wait_ready(
        self, *, health_url: str | None = None, timeout: float = 120.0
    ) -> None:
        """Wait until the tunnel endpoint is reachable.

        If *health_url* is given, polls ``GET <url>`` for HTTP 200.
        Otherwise just checks that the local port is open.
        """
        if health_url:
            self._poll_health(health_url, timeout)
        elif self._local_port:
            self._wait_for_local_port(self._local_port, timeout)

    def check_health(self) -> bool:
        """Return *True* if the SSH process is still alive."""
        return self.is_open

    # Context manager -------------------------------------------------

    def __enter__(self) -> SshTunnel:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # Internals -------------------------------------------------------

    def _build_ssh_cmd(self) -> list[str]:
        cmd = [
            "ssh",
            "-N",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "UserKnownHostsFile=/dev/null",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ServerAliveCountMax=5",
            "-o",
            "ConnectTimeout=15",
            "-o",
            "ExitOnForwardFailure=yes",
            "-o",
            "LogLevel=ERROR",
            "-i",
            self._key_file,
            "-p",
            str(self._port),
        ]
        if self._simple_forward_port is not None:
            cmd += [
                "-L",
                f"127.0.0.1:{self._local_port}:127.0.0.1:{self._simple_forward_port}",
            ]
        for spec in self._forwards:
            cmd += ["-L", spec]
        for spec in self._reverses:
            cmd += ["-R", spec]
        cmd.append(f"{self._user}@{self._host}")
        return cmd

    def _kill(self) -> None:
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            log.info("SSH tunnel closed (pid=%d)", self._proc.pid)
        except ProcessLookupError:
            pass
        finally:
            self._proc = None

    def _wait_for_local_port(self, port: int, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc and self._proc.poll() is not None:
                raise RuntimeError("SSH tunnel process exited while waiting for port")
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(1.0)
                    s.connect(("127.0.0.1", port))
                    return
            except Exception:
                time.sleep(0.3)
        raise TimeoutError(f"Local port 127.0.0.1:{port} not open after {timeout:.0f}s")

    def _poll_health(self, url: str, timeout: float) -> None:
        import urllib.error
        import urllib.request

        deadline = time.monotonic() + timeout
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            if not self.is_open:
                raise RuntimeError("SSH tunnel died while waiting for health endpoint")
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        log.info("Health endpoint ready (attempt %d): %s", attempt, url)
                        return
            except (urllib.error.URLError, OSError, TimeoutError):
                pass
            time.sleep(min(3.0, 1.0 + attempt * 0.5))
        raise TimeoutError(f"Health endpoint not reachable after {timeout:.0f}s: {url}")


# =====================================================================
# SSH sidecar container builder
# =====================================================================


def build_ssh_sidecar_container(
    sidecar_cfg: SshSidecarConfig,
    *,
    max_lifetime_sec: int,
    log_group: str | None = None,
    log_region: str = "us-east-1",
    log_stream_prefix: str = "ecs-sandbox",
) -> dict[str, Any]:
    """Return an ECS container definition dict for the SSH sidecar.

    The sidecar:
    - Installs openssh-server on Alpine (< 2 s).
    - Injects the SSH public key via ECS ``secrets`` from Secrets Manager.
    - Runs sshd as PID 1 (foreground) with a background watchdog for TTL.
    - Has a health check (``nc -z localhost <port>``).
    """
    port = sidecar_cfg.sshd_port
    image = sidecar_cfg.image or "alpine:latest"

    sshd_cfg = (
        f"Port {port}\\n"
        "PermitRootLogin prohibit-password\\n"
        "PasswordAuthentication no\\n"
        "AllowTcpForwarding yes\\n"
        "PermitListen any\\n"
        "GatewayPorts yes\\n"
        "X11Forwarding no\\n"
        "PrintMotd no\\n"
        "LogLevel ERROR\\n"
        "ClientAliveInterval 15\\n"
        "ClientAliveCountMax 3\\n"
        "TCPKeepAlive yes\\n"
        "UseDNS no\\n"
        "MaxSessions 50\\n"
    )

    watchdog = ""
    if max_lifetime_sec > 0:
        watchdog = (
            f"( sleep {max_lifetime_sec}; "
            f"echo 'sidecar watchdog: TTL ({max_lifetime_sec}s) reached'; "
            "kill 1 2>/dev/null; sleep 3; kill -9 1 2>/dev/null ) & "
        )

    sshd_cmd = (
        "set -e; "
        "apk add --no-cache openssh-server netcat-openbsd; "
        "mkdir -p /root/.ssh; chmod 700 /root/.ssh; "
        'printf "%s\\n" "$SSH_PUBLIC_KEY" > /root/.ssh/authorized_keys; '
        "chmod 600 /root/.ssh/authorized_keys; "
        "ssh-keygen -A; "
        f"printf '{sshd_cfg}' > /etc/ssh/sshd_config; "
        f"{watchdog}"
        f"exec /usr/sbin/sshd -D -e -p {port}"
    )

    container: dict[str, Any] = {
        "name": "ssh-tunnel",
        "image": image,
        "essential": True,
        "entryPoint": ["sh", "-c"],
        "command": [sshd_cmd],
        "secrets": [
            {"name": "SSH_PUBLIC_KEY", "valueFrom": sidecar_cfg.public_key_secret_arn},
        ],
        "healthCheck": {
            "command": ["CMD-SHELL", f"nc -z localhost {port} || exit 1"],
            "interval": 5,
            "timeout": 3,
            "retries": 10,
            "startPeriod": 30,
        },
    }

    if log_group:
        container["logConfiguration"] = {
            "logDriver": "awslogs",
            "options": {
                "awslogs-group": log_group,
                "awslogs-region": log_region,
                "awslogs-stream-prefix": f"{log_stream_prefix}-tunnel",
                "awslogs-create-group": "true",
            },
        }

    return container


# =====================================================================
# Exec server — embedded script + HTTP client
# =====================================================================


EXEC_SERVER_SCRIPT = r'''#!/usr/bin/env python3
"""Zero-dependency HTTP exec server for sandbox containers.

Endpoints:
  POST /exec     {"cmd":"...","timeout":300}  -> {"stdout":"...","stderr":"...","rc":0}
  POST /upload   {"path":"/dst","content":"<b64>","mode":"0755"} -> {"ok":true}
  GET  /download?path=/file  -> raw bytes
  GET  /health   -> {"ok":true}

Binds to 127.0.0.1 only (never network-exposed).
"""
from __future__ import annotations
import base64, json, os, subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

_PORT = int(os.environ.get("TB_EXEC_PORT", "19542"))
_BIND = os.environ.get("TB_EXEC_BIND", "127.0.0.1")

class _H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def do_GET(self):
        p = urlparse(self.path)
        if p.path == "/health": self._ok({"ok": True})
        elif p.path == "/download":
            qs = parse_qs(p.query)
            paths = qs.get("path", [])
            if not paths: self._err(400, "missing ?path=")
            else: self._dl(paths[0])
        else: self._err(404, f"not found: {p.path}")
    def do_POST(self):
        p = urlparse(self.path)
        body = self._body()
        if p.path == "/exec": self._exec(body)
        elif p.path == "/upload": self._up(body)
        else: self._err(404, f"not found: {p.path}")
    def _exec(self, b):
        cmd = b.get("cmd")
        if not cmd: self._err(400, "missing 'cmd'"); return
        t = b.get("timeout", 300)
        try:
            cp = subprocess.run(cmd, shell=True, capture_output=True, timeout=t)
            self._ok({"stdout": cp.stdout.decode("utf-8", errors="replace"),
                       "stderr": cp.stderr.decode("utf-8", errors="replace"),
                       "rc": cp.returncode})
        except subprocess.TimeoutExpired:
            self._ok({"stdout":"","stderr":f"timed out after {t}s","rc":124})
        except Exception as e:
            self._ok({"stdout":"","stderr":str(e),"rc":-1})
    def _up(self, b):
        path, c = b.get("path"), b.get("content")
        if not path or c is None: self._err(400, "missing path/content"); return
        try:
            data = base64.b64decode(c)
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "wb") as f: f.write(data)
            m = b.get("mode")
            if m: os.chmod(path, int(m, 8))
            self._ok({"ok": True})
        except Exception as e: self._err(500, str(e))
    def _dl(self, path):
        if not os.path.isfile(path): self._err(404, f"not found: {path}"); return
        try:
            with open(path, "rb") as f: data = f.read()
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers(); self.wfile.write(data)
        except Exception as e: self._err(500, str(e))
    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        if n == 0: return {}
        try: return json.loads(self.rfile.read(n))
        except Exception: return {}
    def _ok(self, obj):
        p = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(p)))
        self.end_headers(); self.wfile.write(p)
    def _err(self, code, msg):
        p = json.dumps({"error": msg}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers(); self.wfile.write(p)

if __name__ == "__main__":
    s = HTTPServer((_BIND, _PORT), _H)
    print(f"exec_server on {_BIND}:{_PORT}", flush=True)
    try: s.serve_forever()
    except KeyboardInterrupt: pass
    finally: s.server_close()
'''

# Transient network errors that the ExecClient should retry.
_TRANSIENT_ERRORS = (
    ConnectionResetError,
    ConnectionRefusedError,
    ConnectionAbortedError,
    BrokenPipeError,
    TimeoutError,
    OSError,
)


class ExecClient:
    """HTTP client for the exec server running inside the container.

    Communicates through the SSH tunnel (``127.0.0.1:<local_port>``).
    Uses only stdlib ``urllib.request`` — no extra dependencies.
    """

    def __init__(self, *, port: int, connect_timeout: float = 30.0) -> None:
        self._base = f"http://127.0.0.1:{port}"
        self._timeout = connect_timeout

    def exec(self, cmd: str, *, timeout: int = 300) -> ExecResult:
        resp = self._post("/exec", {"cmd": cmd, "timeout": timeout})
        return ExecResult(
            stdout=resp.get("stdout", ""),
            stderr=resp.get("stderr", ""),
            return_code=resp.get("rc", -1),
        )

    def upload(
        self,
        remote_path: str,
        data: bytes | Path,
        *,
        mode: str | None = None,
        max_retries: int = 3,
    ) -> None:
        if isinstance(data, Path):
            data = data.read_bytes()
        body: dict[str, Any] = {
            "path": remote_path,
            "content": base64.b64encode(data).decode(),
        }
        if mode is not None:
            body["mode"] = mode
        payload_mb = len(body["content"]) / (1024 * 1024)
        upload_timeout = max(self._timeout, 60.0 + payload_mb * 2.0)
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._post("/upload", body, timeout_override=upload_timeout)
                if not resp.get("ok"):
                    raise RuntimeError(f"upload to {remote_path} failed: {resp}")
                return
            except (TimeoutError, OSError, RuntimeError) as exc:
                last_err = exc
                if attempt < max_retries:
                    log.warning(
                        "upload %s attempt %d/%d: %s",
                        remote_path,
                        attempt,
                        max_retries,
                        exc,
                    )
                    time.sleep(2.0 * attempt)
        raise RuntimeError(
            f"upload to {remote_path} failed after {max_retries} attempts: {last_err}"
        )

    def download(self, remote_path: str, *, max_retries: int = 3) -> bytes:
        import urllib.error
        import urllib.request

        url = f"{self._base}/download?path={urllib.request.quote(remote_path)}"
        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    return resp.read()
            except urllib.error.HTTPError as exc:
                body = exc.read().decode(errors="replace")
                raise RuntimeError(
                    f"download {remote_path} failed (HTTP {exc.code}): {body}"
                ) from exc
            except (*_TRANSIENT_ERRORS, urllib.error.URLError) as exc:
                last_err = exc
                if attempt < max_retries:
                    time.sleep(min(10.0, 2.0 ** (attempt - 1)))
                    continue
                raise ConnectionError(
                    f"download {remote_path} failed: {last_err}"
                ) from last_err
        raise ConnectionError(f"download {remote_path} unreachable")

    def health(self) -> bool:
        try:
            import urllib.request

            req = urllib.request.Request(f"{self._base}/health", method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return resp.status == 200
        except Exception:
            return False

    def _post(
        self,
        path: str,
        body: dict[str, Any],
        *,
        timeout_override: float | None = None,
        max_retries: int = 4,
    ) -> dict[str, Any]:
        import urllib.error
        import urllib.request

        url = f"{self._base}{path}"
        payload = json.dumps(body).encode()

        if timeout_override is not None:
            http_timeout = timeout_override
        else:
            cmd_timeout = body.get("timeout")
            http_timeout = (
                max(self._timeout, cmd_timeout + 30)
                if isinstance(cmd_timeout, (int, float))
                else self._timeout
            )

        last_err: Exception | None = None
        for attempt in range(1, max_retries + 1):
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=http_timeout) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode(errors="replace")
                raise RuntimeError(
                    f"POST {path} failed (HTTP {exc.code}): {raw}"
                ) from exc
            except (*_TRANSIENT_ERRORS, urllib.error.URLError) as exc:
                last_err = exc
                if attempt < max_retries:
                    wait = min(15.0, 2.0 ** (attempt - 1))
                    log.warning(
                        "POST %s attempt %d/%d: %s — retry in %.1fs",
                        path,
                        attempt,
                        max_retries,
                        exc,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                raise ConnectionError(
                    f"POST {path} failed after {max_retries} attempts: {last_err}"
                ) from last_err
        raise ConnectionError(f"POST {path} unreachable")


# =====================================================================
# Image builder — AWS CodeBuild + ECR caching
# =====================================================================


class ImageBuilder:
    """Build Docker images via AWS CodeBuild and push to ECR.

    Features:
    - Content-hash based ECR tags for automatic caching.
    - Build deduplication across concurrent tasks (only one build per tag).
    - Semaphore-based concurrency control.
    """

    _lock = threading.Lock()
    _inflight_builds: dict[str, threading.Event] = {}
    _build_semaphore: threading.Semaphore | None = None
    _build_semaphore_size: int = 0

    @staticmethod
    def get_ecr_image_tag(environment_dir: str | Path, environment_name: str) -> str:
        """``<name>__<content_hash[:8]>`` — deterministic, cache-friendly."""
        from dirhash import dirhash as _dirhash

        content_hash = _dirhash(str(environment_dir), "sha256")[:8]
        return f"{environment_name}__{content_hash}"

    @staticmethod
    def image_exists_in_ecr(
        ecr_repository: str, tag: str, region: str | None = None
    ) -> bool:
        boto3, _, ClientError = _require_aws_sdks()
        ecr = boto3.client("ecr", region_name=region)
        repo_name = (
            ecr_repository.split("/", 1)[1] if "/" in ecr_repository else ecr_repository
        )
        try:
            ecr.describe_images(repositoryName=repo_name, imageIds=[{"imageTag": tag}])
            return True
        except ClientError:
            return False

    @classmethod
    def ensure_image_built(
        cls,
        *,
        cfg: EcsFargateConfig,
        environment_name: str,
        force_build: bool = False,
    ) -> str:
        """Build and push if needed.  Returns the full ECR image URL.

        Safe to call from many threads — deduplication and a semaphore
        ensure only one CodeBuild job runs per content-hash tag.
        """
        ecr_repo = cfg.ecr_repository
        env_dir = cfg.environment_dir
        if not ecr_repo or not env_dir:
            raise ValueError(
                "ecr_repository and environment_dir are required for image building"
            )

        tag = cls.get_ecr_image_tag(env_dir, environment_name)
        image_url = f"{ecr_repo}:{tag}"

        # --- Dedup: check if another thread is already building this tag ---
        with cls._lock:
            if tag in cls._inflight_builds:
                event = cls._inflight_builds[tag]
                log.info("Build already in progress for %s — waiting", tag)
                waiting = True
            else:
                waiting = False

        if waiting:
            event.wait()
            log.info("Build finished (by another thread): %s", tag)
            return image_url

        # --- ECR cache check ---
        if not force_build and cls.image_exists_in_ecr(ecr_repo, tag, cfg.region):
            log.info("ECR cache hit — skipping build: %s", image_url)
            return image_url

        # --- Register as builder ---
        event = threading.Event()
        with cls._lock:
            if tag in cls._inflight_builds:
                existing = cls._inflight_builds[tag]
            else:
                cls._inflight_builds[tag] = event
                existing = None

        if existing is not None:
            log.info("Build started by another thread — joining")
            existing.wait()
            return image_url

        # --- Initialise semaphore ---
        with cls._lock:
            if (
                cls._build_semaphore is None
                or cls._build_semaphore_size != cfg.build_parallelism
            ):
                cls._build_semaphore = threading.Semaphore(cfg.build_parallelism)
                cls._build_semaphore_size = cfg.build_parallelism

        try:
            log.info(
                "Waiting for CodeBuild slot (parallelism=%d)", cfg.build_parallelism
            )
            cls._build_semaphore.acquire()  # type: ignore[union-attr]
            try:
                if not force_build and cls.image_exists_in_ecr(
                    ecr_repo, tag, cfg.region
                ):
                    log.info("ECR cache hit (after slot): %s", image_url)
                    return image_url
                cls._build_and_push(
                    cfg=cfg,
                    environment_name=environment_name,
                    tag=tag,
                    image_url=image_url,
                )
            finally:
                cls._build_semaphore.release()  # type: ignore[union-attr]
        finally:
            event.set()
            with cls._lock:
                cls._inflight_builds.pop(tag, None)

        return image_url

    @classmethod
    def _build_and_push(
        cls,
        *,
        cfg: EcsFargateConfig,
        environment_name: str,
        tag: str,
        image_url: str,
    ) -> None:
        boto3, _, ClientError = _require_aws_sdks()
        ecr_repo = cfg.ecr_repository or ""
        ecr_registry = ecr_repo.split("/")[0]
        repo_name = ecr_repo.split("/", 1)[1] if "/" in ecr_repo else ecr_repo
        env_dir = Path(cfg.environment_dir or ".")

        log.info("Building image via CodeBuild: %s", image_url)

        # 1. Upload build context (ZIP) to S3
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in env_dir.rglob("*"):
                if item.is_file():
                    zf.write(item, arcname=str(item.relative_to(env_dir)))
        buf.seek(0)

        s3 = boto3.client("s3", region_name=cfg.region)
        s3_prefix = cfg.s3_prefix or "ecs-sandbox"
        nonce = uuid.uuid4().hex[:8]
        s3_key = f"{s3_prefix}/codebuild/{environment_name}-{nonce}.zip"
        s3.put_object(Bucket=cfg.s3_bucket, Key=s3_key, Body=buf.read())

        # 2. Resolve or create CodeBuild project
        cb = boto3.client("codebuild", region_name=cfg.region)
        if cfg.codebuild_project:
            project_name = cfg.codebuild_project
        else:
            if not cfg.codebuild_service_role:
                raise RuntimeError(
                    "codebuild_project or codebuild_service_role is required"
                )
            project_name = f"ecs-sandbox-build-{nonce}"
            try:
                cb.create_project(
                    name=project_name,
                    source={"type": "NO_SOURCE", "buildspec": "version: 0.2"},
                    artifacts={"type": "NO_ARTIFACTS"},
                    environment={
                        "type": "LINUX_CONTAINER",
                        "image": "aws/codebuild/amazonlinux-x86_64-standard:5.0",
                        "computeType": cfg.codebuild_compute_type,
                        "privilegedMode": True,
                    },
                    serviceRole=cfg.codebuild_service_role,
                    timeoutInMinutes=cfg.codebuild_build_timeout,
                )
            except ClientError as e:
                if "already exists" not in str(e).lower():
                    raise

        # 3. Inline buildspec
        pre_build_cmds = [
            f"aws ecr get-login-password --region $AWS_DEFAULT_REGION"
            f" | docker login --username AWS --password-stdin {ecr_registry}",
        ]
        if cfg.dockerhub_secret_arn:
            pre_build_cmds.append(
                f"DOCKERHUB_CREDS=$(aws secretsmanager get-secret-value"
                f" --secret-id {cfg.dockerhub_secret_arn}"
                f" --query SecretString --output text --region $AWS_DEFAULT_REGION)"
                f' && echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;c=json.load(sys.stdin);print(c['password'])" """
                f'| docker login -u $(echo "$DOCKERHUB_CREDS" | python3 -c'
                """ "import sys,json;print(json.load(sys.stdin)['username'])") """
                f"--password-stdin"
            )
        pre_yaml = "\n".join(f"      - {c}" for c in pre_build_cmds)
        build_cmd = (
            f"for i in 1 2 3; do docker build -t {repo_name}:{tag} . && break; "
            f'echo "build failed ($i/3), retry in 30s"; sleep 30; done'
        )
        buildspec = (
            "version: 0.2\nphases:\n  pre_build:\n    commands:\n"
            f"{pre_yaml}\n  build:\n    commands:\n"
            f"      - {build_cmd}\n      - docker tag {repo_name}:{tag} {image_url}\n"
            f"  post_build:\n    commands:\n      - docker push {image_url}\n"
        )

        # 4. Start build
        resp = cb.start_build(
            projectName=project_name,
            sourceTypeOverride="S3",
            sourceLocationOverride=f"{cfg.s3_bucket}/{s3_key}",
            buildspecOverride=buildspec,
            timeoutInMinutesOverride=cfg.codebuild_build_timeout,
            privilegedModeOverride=True,
            environmentTypeOverride="LINUX_CONTAINER",
            imageOverride="aws/codebuild/amazonlinux-x86_64-standard:5.0",
            computeTypeOverride=cfg.codebuild_compute_type,
        )
        build_id = resp["build"]["id"]
        log.info("CodeBuild started: %s", build_id)

        # 5. Poll until complete
        while True:
            time.sleep(10)
            status_resp = cb.batch_get_builds(ids=[build_id])
            build = status_resp["builds"][0]
            status = build["buildStatus"]
            if status == "SUCCEEDED":
                log.info("CodeBuild succeeded: %s", build_id)
                return
            if status in ("FAILED", "FAULT", "STOPPED", "TIMED_OUT"):
                phases = build.get("phases", [])
                failed = [
                    p for p in phases if p.get("phaseStatus") not in (None, "SUCCEEDED")
                ]
                ctx = (
                    "; ".join(
                        f"{p['phaseType']}: {p.get('phaseStatus')}" for p in failed
                    )
                    or status
                )
                raise RuntimeError(
                    f"CodeBuild failed for {image_url}: {ctx} (build: {build_id})"
                )
            log.debug(
                "CodeBuild %s — phase=%s status=%s",
                build_id,
                build.get("currentPhase"),
                status,
            )


# =====================================================================
# Core sandbox — ECS task lifecycle + SSH connectivity
# =====================================================================

_active_sandboxes: dict[int, Any] = {}
_cleanup_lock = threading.Lock()
_atexit_registered = False
_PROCESS_NONCE = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
_exec_server_url_cache: dict[str, str] = {}


def _emergency_cleanup() -> None:
    with _cleanup_lock:
        for sb in list(_active_sandboxes.values()):
            try:
                sb.stop()
            except Exception:
                pass


class EcsFargateSandbox:
    """ECS Fargate sandbox implementing the :class:`Sandbox` protocol.

    Supports two modes (determined by ``ssh_sidecar.exec_server_port``):

    **Exec-server mode** (``exec_server_port`` is set):
      One-way SSH tunnel + embedded HTTP exec server.  ``exec()``,
      ``upload()``, ``download()`` work through the exec server.

    **Agent-server mode** (``exec_server_port`` is ``None``):
      Two-way SSH tunnel.  Consumer accesses the hosted agent via
      ``ssh_tunnel`` / ``local_port``.  ``exec()`` / ``upload()`` /
      ``download()`` raise :class:`RuntimeError`.
    """

    def __init__(self, cfg: EcsFargateConfig, *, task_id: str, run_id: str) -> None:
        self._cfg = cfg
        self._task_id = task_id
        self._run_id = run_id
        self._task_arn: str | None = None
        self._task_def_arn: str | None = None
        self._task_ip: str | None = None
        self._ssh_key_file: str | None = None
        self._ssh_tunnel: SshTunnel | None = None
        self._exec_client: ExecClient | None = None
        self._started = False
        self._stopped = False
        self._ecs: Any = None
        self._ec2: Any = None
        # For agent-server mode (two-way tunnel)
        self._ssh_tunnel_port: int | None = None
        self._agent_forward_port: int | None = None

    # Public API -------------------------------------------------------

    @property
    def task_arn(self) -> str | None:
        return self._task_arn

    @property
    def task_ip(self) -> str | None:
        return self._task_ip

    @property
    def local_port(self) -> int | None:
        """Local port of the SSH forward tunnel (exec server or agent server)."""
        if self._ssh_tunnel:
            try:
                return self._ssh_tunnel.local_port
            except RuntimeError:
                pass
        return None

    @property
    def ssh_tunnel(self) -> SshTunnel | None:
        return self._ssh_tunnel

    @property
    def exec_client(self) -> ExecClient | None:
        """The exec client (only available in exec-server mode after start)."""
        return self._exec_client

    @property
    def model_tunnel_port(self) -> int | None:
        """Port the container uses to reach the model (agent-server mode only)."""
        return self._ssh_tunnel_port

    @property
    def is_running(self) -> bool:
        return self._started and not self._stopped

    def reconnect_tunnel(self) -> None:
        """Re-open the SSH tunnel if it died (e.g. after a network blip)."""
        if self._stopped or not self._started:
            raise RuntimeError("Cannot reconnect tunnel on a stopped/unstarted sandbox")
        sidecar = self._cfg.ssh_sidecar
        if sidecar is None:
            return
        if self._ssh_tunnel:
            self._ssh_tunnel.close()
            self._ssh_tunnel = None
        self._open_tunnel(sidecar)

    def start(self, *, force_build: bool = False) -> None:
        if self._started:
            return
        try:
            self._do_start(force_build=force_build)
            self._started = True
            self._register_for_cleanup()
        except Exception:
            self._cleanup()
            raise

    def describe_task(self) -> dict[str, Any] | None:
        """Return a summary dict of the ECS task's current state, or None."""
        if self._task_arn is None or self._ecs is None:
            return None
        try:
            resp = self._ecs.describe_tasks(
                cluster=self._cfg.cluster, tasks=[self._task_arn]
            )
            tasks = resp.get("tasks") or []
            if not tasks:
                return {"taskArn": self._task_arn}
            t = tasks[0]
            return {
                "taskArn": t.get("taskArn"),
                "lastStatus": t.get("lastStatus"),
                "desiredStatus": t.get("desiredStatus"),
                "stopCode": t.get("stopCode"),
                "stoppedReason": t.get("stoppedReason"),
            }
        except Exception as e:
            return {"taskArn": self._task_arn, "error": str(e)}

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._cleanup()
        self._unregister_from_cleanup()

    def exec(self, command: str, timeout_sec: float = 180) -> ExecResult:
        self._require_exec_client()
        return self._exec_client.exec(command, timeout=int(timeout_sec))  # type: ignore[union-attr]

    def upload(self, local_path: Path, remote_path: str) -> None:
        self._require_exec_client()
        local = Path(local_path)
        if local.is_dir():
            for child in local.rglob("*"):
                if child.is_file():
                    rel = child.relative_to(local)
                    self.upload(child, f"{remote_path}/{rel}")
            return
        if local.stat().st_size > 512 * 1024 and self._cfg.s3_bucket:
            self._upload_via_s3([local], os.path.dirname(remote_path) or "/tmp")
        else:
            self._exec_client.upload(remote_path, local)  # type: ignore[union-attr]

    def download(self, remote_path: str, local_path: Path) -> None:
        self._require_exec_client()
        data = self._exec_client.download(remote_path)  # type: ignore[union-attr]
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)

    def __enter__(self) -> EcsFargateSandbox:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # Start implementation --------------------------------------------

    def _do_start(self, *, force_build: bool) -> None:
        cfg = self._cfg
        sidecar = cfg.ssh_sidecar
        if sidecar is None:
            raise ValueError("ssh_sidecar must be configured")

        # 0. Init boto3 clients
        self._init_aws_clients()

        # 1. Docker build (if configured)
        built_image: str | None = None
        if cfg.ecr_repository and cfg.environment_dir:
            built_image = ImageBuilder.ensure_image_built(
                cfg=cfg,
                environment_name=_sanitize_id(self._task_id),
                force_build=force_build,
            )

        # 2. Resolve image
        image = self._resolve_image(built_image)

        # 3. Download SSH private key
        if not sidecar.private_key_secret_arn:
            raise ValueError(
                "ssh_sidecar.private_key_secret_arn is required (pre-provisioned keys only)"
            )
        self._ssh_key_file = download_secret_to_file(
            sidecar.private_key_secret_arn, cfg.region
        )

        # 4. Resolve tunnel port for agent-server mode
        has_exec_server = sidecar.exec_server_port is not None
        if not has_exec_server:
            self._ssh_tunnel_port = self._resolve_ssh_tunnel_port(sidecar)

        # 5. Upload exec server to S3 (exec-server mode only)
        exec_server_url: str | None = None
        if has_exec_server:
            exec_server_url = self._upload_exec_server()

        # 6. Build container command
        command = self._build_container_command(exec_server_url, sidecar)

        # 7. Build environment variables
        env = self._build_env_vars(sidecar)

        # 8. Build sidecar container
        log_region = cfg.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        sidecar_def = build_ssh_sidecar_container(
            sidecar,
            max_lifetime_sec=cfg.max_task_lifetime_sec,
            log_group=cfg.log_group,
            log_region=log_region,
            log_stream_prefix=cfg.log_stream_prefix or "ecs-sandbox",
        )

        # 9. Register task definition
        self._task_def_arn = self._register_task_definition(
            image=image,
            command=command,
            env=env,
            sidecar_def=sidecar_def,
        )

        # 10. Run task
        self._task_arn = self._run_task(self._task_def_arn)

        # 11. Wait for RUNNING
        self._wait_for_running()

        # 12. Get task IP
        self._task_ip = self._get_task_public_ip()

        # 13. Wait for SSH ready
        self._wait_for_ssh_ready(
            self._task_ip, sidecar.sshd_port, sidecar.ssh_ready_timeout_sec
        )

        # 14. Open SSH tunnel
        self._open_tunnel(sidecar)

        # 15. Wait for readiness
        if has_exec_server:
            health_url = f"http://127.0.0.1:{self._ssh_tunnel.local_port}/health"  # type: ignore[union-attr]
            self._ssh_tunnel.wait_ready(
                health_url=health_url, timeout=sidecar.ssh_ready_timeout_sec
            )  # type: ignore[union-attr]
            self._exec_client = ExecClient(port=self._ssh_tunnel.local_port)  # type: ignore[union-attr]

    # Internal methods ------------------------------------------------

    def _init_aws_clients(self) -> None:
        boto3, Config, _ = _require_aws_sdks()
        boto_cfg = Config(
            connect_timeout=30,
            read_timeout=60,
            retries={"max_attempts": 8, "mode": "adaptive"},
        )
        self._ecs = boto3.client("ecs", region_name=self._cfg.region, config=boto_cfg)
        self._ec2 = boto3.client("ec2", region_name=self._cfg.region, config=boto_cfg)

    def _resolve_image(self, built_image: str | None = None) -> str:
        if built_image:
            return built_image
        cfg = self._cfg
        if cfg.image_template:
            sanitized = _sanitize_id(self._task_id)
            return cfg.image_template.format(
                task_id=self._task_id, task_id_sanitized=sanitized
            )
        return ""

    @staticmethod
    def _resolve_ssh_tunnel_port(sidecar: SshSidecarConfig) -> int:
        if sidecar.local_port > 0:
            return sidecar.local_port
        for env_name in sidecar.target_url_env.split(","):
            env_name = env_name.strip()
            if not env_name:
                continue
            env_value = os.getenv(env_name)
            if not env_value:
                continue
            parsed = urlparse(env_value)
            if parsed.port:
                return int(parsed.port)
        raise ValueError(
            f"ssh_sidecar.local_port is 0 and no port could be inferred from "
            f"env vars: {sidecar.target_url_env}"
        )

    @staticmethod
    def _resolve_tunnel_target(sidecar: SshSidecarConfig) -> tuple[str, int]:
        for env_name in sidecar.target_url_env.split(","):
            env_name = env_name.strip()
            if not env_name:
                continue
            env_value = os.getenv(env_name)
            if not env_value:
                continue
            parsed = urlparse(env_value)
            host = parsed.hostname
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if host:
                return host, port
        raise ValueError(f"Cannot resolve tunnel target from: {sidecar.target_url_env}")

    def _upload_exec_server(self) -> str:
        cfg = self._cfg
        if not cfg.s3_bucket:
            raise ValueError("s3_bucket is required for exec server upload")
        cache_key = f"{cfg.s3_bucket}/{self._run_id}"
        if cache_key in _exec_server_url_cache:
            return _exec_server_url_cache[cache_key]

        boto3, *_ = _require_aws_sdks()
        s3 = boto3.client("s3", region_name=cfg.region)
        prefix = cfg.s3_prefix or "ecs-sandbox"
        key = f"{prefix}/{self._run_id}-{_PROCESS_NONCE}/_exec_server/exec_server.py"
        s3.put_object(Bucket=cfg.s3_bucket, Key=key, Body=EXEC_SERVER_SCRIPT.encode())
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg.s3_bucket, "Key": key},
            ExpiresIn=21600,
        )
        _exec_server_url_cache[cache_key] = url
        log.info("Uploaded exec server → s3://%s/%s", cfg.s3_bucket, key)
        return url

    def _build_container_command(
        self,
        exec_server_url: str | None,
        sidecar: SshSidecarConfig,
    ) -> list[str] | None:
        """Return the main container entrypoint, or None to keep the image default."""
        if exec_server_url is None:
            return None  # agent-server mode — image has its own CMD
        exec_port = sidecar.exec_server_port or 19542
        ttl = self._cfg.max_task_lifetime_sec
        hostname = re.sub(r"[^A-Za-z0-9._-]", "-", self._task_id)[:63]
        bootstrap = (
            "if command -v python >/dev/null 2>&1; then :; "
            "elif command -v python3 >/dev/null 2>&1; then "
            '  P=$(command -v python3); ln -sf "$P" /usr/local/bin/python 2>/dev/null || true; '
            '  ln -sf "$P" /usr/bin/python 2>/dev/null || true; fi; '
        )
        setup = (
            f"{bootstrap}"
            f"hostname {shlex.quote(hostname)} 2>/dev/null || true; "
            f"python3 -c 'import urllib.request as u,sys;"
            f'u.urlretrieve(sys.argv[1],"/tmp/_exec_server.py")\' '
            f"{shlex.quote(exec_server_url)} && "
            f"TB_EXEC_PORT={exec_port} TB_EXEC_BIND=127.0.0.1 "
            f"nohup python3 /tmp/_exec_server.py >/tmp/_exec.log 2>&1 & "
            f"sleep {ttl}"
        )
        return ["sh", "-lc", setup]

    def _build_env_vars(self, sidecar: SshSidecarConfig) -> dict[str, str]:
        env: dict[str, str] = {}
        cfg = self._cfg
        if cfg.extra_env:
            for k, v in cfg.extra_env.items():
                env[k] = self._render_env_value(v)
        if cfg.model_endpoint_env:
            env[cfg.model_endpoint_env.name] = self._render_env_value(
                cfg.model_endpoint_env.value
            )
        if self._ssh_tunnel_port and sidecar.model_env_var:
            env[sidecar.model_env_var] = f"http://127.0.0.1:{self._ssh_tunnel_port}"
        return env

    def _render_env_value(self, value: str) -> str:
        if self._ssh_tunnel_port is not None:
            value = value.replace("{ssh_tunnel_port}", str(self._ssh_tunnel_port))
        if self._task_ip:
            value = value.replace("{task_ip}", self._task_ip)
        value = value.replace("{task_id}", self._task_id)
        return value

    def _register_task_definition(
        self,
        *,
        image: str,
        command: list[str] | None,
        env: dict[str, str],
        sidecar_def: dict[str, Any],
    ) -> str:
        cfg = self._cfg
        log_region = cfg.region or os.environ.get("AWS_DEFAULT_REGION", "us-east-1")

        log_cfg: dict[str, Any] | None = None
        if cfg.log_group:
            log_cfg = {
                "logDriver": "awslogs",
                "options": {
                    "awslogs-group": cfg.log_group,
                    "awslogs-region": log_region,
                    "awslogs-stream-prefix": cfg.log_stream_prefix or "ecs-sandbox",
                    "awslogs-create-group": "true",
                },
            }

        # Try cloning from base task definition
        base: dict[str, Any] | None = None
        if cfg.task_definition:
            try:
                resp = self._ecs.describe_task_definition(
                    taskDefinition=cfg.task_definition
                )
                base = resp["taskDefinition"]
            except Exception:
                base = None

        if base is not None:
            return self._register_from_base(
                base=base,
                image=image,
                command=command,
                env=env,
                sidecar_def=sidecar_def,
                log_cfg=log_cfg,
            )
        return self._register_from_scratch(
            image=image,
            command=command,
            env=env,
            sidecar_def=sidecar_def,
            log_cfg=log_cfg,
        )

    def _register_from_base(
        self,
        *,
        base: dict,
        image: str,
        command: list[str] | None,
        env: dict[str, str],
        sidecar_def: dict,
        log_cfg: dict | None,
    ) -> str:
        cfg = self._cfg
        containers = list(base.get("containerDefinitions") or [])

        target = None
        for cd in containers:
            if cd.get("name") == cfg.container_name:
                target = cd
                break
        if target is None:
            names = [c.get("name") for c in containers]
            raise RuntimeError(
                f"Base task-def has no container '{cfg.container_name}'. Available: {names}"
            )

        if image:
            target["image"] = image
        if command is not None:
            target["command"] = command
        if env:
            existing = {e["name"]: e["value"] for e in target.get("environment", [])}
            existing.update(env)
            target["environment"] = [
                {"name": k, "value": v} for k, v in sorted(existing.items())
            ]
        if log_cfg:
            target["logConfiguration"] = log_cfg
        target["dependsOn"] = [{"containerName": "ssh-tunnel", "condition": "HEALTHY"}]

        containers = [c for c in containers if c.get("name") != "ssh-tunnel"]
        containers.append(sidecar_def)

        family = self._make_family_name()
        payload: dict[str, Any] = {
            "family": family,
            "networkMode": base.get("networkMode", "awsvpc"),
            "requiresCompatibilities": base.get("requiresCompatibilities", ["FARGATE"]),
            "cpu": str(max(int(base.get("cpu") or "256"), int(cfg.cpu))),
            "memory": str(max(int(base.get("memory") or "512"), int(cfg.memory))),
            "containerDefinitions": containers,
        }
        eph = max(
            (base.get("ephemeralStorage") or {}).get("sizeInGiB", 20),
            cfg.ephemeral_storage_gib or 20,
        )
        payload["ephemeralStorage"] = {"sizeInGiB": eph}
        for k in ("taskRoleArn", "executionRoleArn", "runtimePlatform", "volumes"):
            if base.get(k) is not None:
                payload[k] = base[k]
        if cfg.execution_role_arn:
            payload["executionRoleArn"] = cfg.execution_role_arn
        if cfg.task_role_arn:
            payload["taskRoleArn"] = cfg.task_role_arn

        return self._do_register(payload)

    def _register_from_scratch(
        self,
        *,
        image: str,
        command: list[str] | None,
        env: dict[str, str],
        sidecar_def: dict,
        log_cfg: dict | None,
    ) -> str:
        cfg = self._cfg
        if not cfg.execution_role_arn:
            raise RuntimeError(
                "execution_role_arn is required when no base task definition is provided"
            )

        container_def: dict[str, Any] = {
            "name": cfg.container_name,
            "essential": True,
            "dependsOn": [{"containerName": "ssh-tunnel", "condition": "HEALTHY"}],
        }
        if image:
            container_def["image"] = image
        if command is not None:
            container_def["command"] = command
        if cfg.container_port:
            container_def["portMappings"] = [
                {"containerPort": cfg.container_port, "protocol": "tcp"}
            ]
        if env:
            container_def["environment"] = [
                {"name": k, "value": v} for k, v in sorted(env.items())
            ]
        if log_cfg:
            container_def["logConfiguration"] = log_cfg

        family = self._make_family_name()
        payload: dict[str, Any] = {
            "family": family,
            "networkMode": "awsvpc",
            "requiresCompatibilities": ["FARGATE"],
            "cpu": cfg.cpu,
            "memory": cfg.memory,
            "executionRoleArn": cfg.execution_role_arn,
            "containerDefinitions": [container_def, sidecar_def],
        }
        if cfg.task_role_arn:
            payload["taskRoleArn"] = cfg.task_role_arn
        if cfg.ephemeral_storage_gib:
            payload["ephemeralStorage"] = {"sizeInGiB": cfg.ephemeral_storage_gib}
        if cfg.platform_version:
            payload["platformVersion"] = cfg.platform_version

        return self._do_register(payload)

    def _do_register(self, payload: dict[str, Any]) -> str:
        max_retries = 25
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._ecs.register_task_definition(**payload)
                arn = resp["taskDefinition"]["taskDefinitionArn"]
                log.info("Registered task def: %s", arn)
                return arn
            except Exception as exc:
                if not is_retryable_error(exc) or attempt >= max_retries:
                    raise RuntimeError(
                        f"register_task_definition failed: {exc}"
                    ) from exc
                delay = min(60.0, 2.0 ** min(6, attempt - 1)) + random.random() * 2
                log.warning(
                    "register_task_definition throttled (%d/%d), retry in %.1fs",
                    attempt,
                    max_retries,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError("register_task_definition failed after max retries")

    def _make_family_name(self) -> str:
        raw = f"{self._cfg.task_definition_family_prefix}-{_sanitize_id(self._task_id)}-{int(time.time())}"
        family = re.sub(r"[^A-Za-z0-9_-]", "_", raw)[:255]
        if not family or not re.match(r"^[A-Za-z0-9]", family):
            family = f"ecs_{family}"
        return family

    def _run_task(self, task_def_arn: str) -> str:
        cfg = self._cfg
        run_kwargs: dict[str, Any] = {
            "cluster": cfg.cluster,
            "taskDefinition": task_def_arn,
            "launchType": "FARGATE",
            "networkConfiguration": {
                "awsvpcConfiguration": {
                    "subnets": cfg.subnets,
                    "securityGroups": cfg.security_groups,
                    "assignPublicIp": "ENABLED" if cfg.assign_public_ip else "DISABLED",
                }
            },
        }
        if cfg.platform_version:
            run_kwargs["platformVersion"] = cfg.platform_version

        last_failures: Any = None
        for attempt in range(1, cfg.run_task_max_retries + 1):
            try:
                resp = retry_with_backoff(
                    lambda: self._ecs.run_task(**run_kwargs),
                    operation_name="run_task",
                )
            except Exception as exc:
                if not is_retryable_error(exc) or attempt >= cfg.run_task_max_retries:
                    raise
                delay = min(60.0, 2.0 ** min(6, attempt - 1)) + random.random() * 2
                log.warning(
                    "run_task failed (%d/%d): %s — retry in %.1fs",
                    attempt,
                    cfg.run_task_max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)
                continue

            failures = resp.get("failures") or []
            if not failures:
                tasks = resp.get("tasks") or []
                if not tasks:
                    raise RuntimeError("run_task returned no tasks")
                task_arn = tasks[0]["taskArn"]
                log.info("Started ECS task: %s", task_arn)
                return task_arn

            last_failures = failures
            reasons = " | ".join(str(f.get("reason", "")) for f in failures)
            if (
                not any(m in reasons for m in _RETRYABLE_MESSAGES)
                or attempt >= cfg.run_task_max_retries
            ):
                raise RuntimeError(f"run_task failures: {failures}")
            delay = min(60.0, 2.0 ** min(6, attempt - 1)) + random.random() * 2
            log.warning(
                "run_task capacity issue (%d/%d): %s — retry in %.1fs",
                attempt,
                cfg.run_task_max_retries,
                reasons,
                delay,
            )
            time.sleep(delay)

        raise RuntimeError(
            f"run_task failed after {cfg.run_task_max_retries} retries: {last_failures}"
        )

    def _wait_for_running(self) -> None:
        cfg = self._cfg
        start = time.time()
        poll = 5.0
        last_status = ""

        while True:
            elapsed = time.time() - start
            if elapsed > cfg.startup_timeout_sec:
                raise TimeoutError(
                    f"ECS task not RUNNING after {elapsed:.0f}s (last: {last_status})"
                )

            try:
                resp = self._ecs.describe_tasks(
                    cluster=cfg.cluster, tasks=[self._task_arn]
                )
            except Exception as exc:
                if is_retryable_error(exc):
                    time.sleep(poll + random.random() * 3)
                    continue
                raise

            tasks = resp.get("tasks") or []
            if not tasks:
                raise RuntimeError("ECS task disappeared")

            status = tasks[0].get("lastStatus", "UNKNOWN")
            if status == "RUNNING":
                log.info("ECS task RUNNING after %.0fs", elapsed)
                return
            if status == "STOPPED":
                raise RuntimeError(f"ECS task stopped: {tasks[0].get('stoppedReason')}")

            if status != last_status:
                log.info("ECS task %s (%.0fs)", status, elapsed)
                last_status = status

            time.sleep(poll + random.random() * 3)
            poll = min(15.0, poll + 0.5)

    def _get_task_public_ip(self) -> str:
        """Resolve the task's public IP from its ENI."""
        max_retries = 10
        for attempt in range(1, max_retries + 1):
            try:
                resp = self._ecs.describe_tasks(
                    cluster=self._cfg.cluster, tasks=[self._task_arn]
                )
                tasks = resp.get("tasks") or []
                if not tasks:
                    raise RuntimeError("Task not found")

                eni_id = None
                for att in tasks[0].get("attachments") or []:
                    if att.get("type") == "ElasticNetworkInterface":
                        for d in att.get("details") or []:
                            if d.get("name") == "networkInterfaceId":
                                eni_id = d["value"]
                                break
                    if eni_id:
                        break

                if not eni_id:
                    for att in tasks[0].get("attachments") or []:
                        for d in att.get("details") or []:
                            if d.get("name") == "privateIPv4Address" and d.get("value"):
                                return d["value"]
                    raise RuntimeError("No ENI/IP yet")

                eni = self._ec2.describe_network_interfaces(
                    NetworkInterfaceIds=[eni_id]
                )
                iface = eni["NetworkInterfaces"][0]
                pub = (iface.get("Association") or {}).get("PublicIp")
                if pub:
                    log.info("Container public IP: %s", pub)
                    return pub
                priv = iface.get("PrivateIpAddress")
                if priv:
                    log.info("Container private IP: %s", priv)
                    return priv
                raise RuntimeError(f"ENI {eni_id} has no IP")

            except Exception as exc:
                if attempt >= max_retries:
                    raise
                if is_retryable_error(exc):
                    time.sleep(min(15.0, 2.0**attempt + random.random()))
                else:
                    log.warning(
                        "get_task_ip attempt %d/%d: %s", attempt, max_retries, exc
                    )
                    time.sleep(min(15.0, 3.0 + attempt * 2))

        raise RuntimeError("get_task_ip exhausted retries")

    @staticmethod
    def _wait_for_ssh_ready(host: str, port: int, timeout: float) -> None:
        deadline = time.time() + timeout
        log.info("Waiting for SSH at %s:%d", host, port)
        while time.time() < deadline:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(5.0)
                    s.connect((host, port))
                    s.settimeout(5.0)
                    data = s.recv(256)
                    if data and b"SSH" in data:
                        log.info("SSH ready at %s:%d", host, port)
                        return
            except Exception:
                pass
            time.sleep(2.0)
        raise TimeoutError(f"SSH not ready at {host}:{port} after {timeout:.0f}s")

    def _open_tunnel(self, sidecar: SshSidecarConfig) -> None:
        assert self._task_ip is not None
        assert self._ssh_key_file is not None

        has_exec_server = sidecar.exec_server_port is not None

        if has_exec_server:
            self._ssh_tunnel = SshTunnel(
                host=self._task_ip,
                port=sidecar.sshd_port,
                user="root",
                key_file=self._ssh_key_file,
                forward_port=sidecar.exec_server_port,
            )
            self._ssh_tunnel.open()
        else:
            remote_host, remote_port = self._resolve_tunnel_target(sidecar)
            local_port = self._ssh_tunnel_port
            assert local_port is not None

            self._agent_forward_port = _free_port()
            container_port = self._cfg.container_port
            if not container_port:
                raise ValueError("container_port is required in agent-server mode")

            forwards = [f"{self._agent_forward_port}:localhost:{container_port}"]
            reverses = [f"{local_port}:{remote_host}:{remote_port}"]

            self._ssh_tunnel = SshTunnel(
                host=self._task_ip,
                port=sidecar.sshd_port,
                user="root",
                key_file=self._ssh_key_file,
                forwards=forwards,
                reverses=reverses,
                local_port_override=self._agent_forward_port,
            )
            self._ssh_tunnel.open()

    # Cleanup ---------------------------------------------------------

    def _cleanup(self) -> None:
        if self._ssh_tunnel:
            try:
                self._ssh_tunnel.close()
            except Exception:
                pass
            self._ssh_tunnel = None

        if self._task_arn and self._ecs:
            try:
                retry_with_backoff(
                    lambda: self._ecs.stop_task(
                        cluster=self._cfg.cluster,
                        task=self._task_arn,
                        reason="sandbox cleanup",
                    ),
                    operation_name="stop_task",
                    max_retries=10,
                )
                log.info("Stopped ECS task: %s", self._task_arn)
            except Exception as exc:
                log.warning("Failed to stop task %s: %s", self._task_arn, exc)

        if self._task_def_arn and self._ecs:
            try:
                retry_with_backoff(
                    lambda: self._ecs.deregister_task_definition(
                        taskDefinition=self._task_def_arn
                    ),
                    operation_name="deregister_task_definition",
                    max_retries=5,
                )
                log.info("Deregistered task def: %s", self._task_def_arn)
            except Exception as exc:
                log.warning(
                    "Failed to deregister task def %s: %s", self._task_def_arn, exc
                )

        if self._ssh_key_file:
            try:
                os.remove(self._ssh_key_file)
            except Exception:
                pass
            self._ssh_key_file = None

    def _require_exec_client(self) -> None:
        if self._exec_client is None:
            raise RuntimeError(
                "exec()/upload()/download() require ssh_sidecar.exec_server_port to be set "
                "(exec-server mode). In agent-server mode, use sandbox.ssh_tunnel directly."
            )

    def _upload_via_s3(self, paths: list[Path], dest_dir: str) -> None:
        cfg = self._cfg
        if not cfg.s3_bucket:
            raise ValueError("s3_bucket is required for S3 staging")

        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for p in paths:
                if p.is_file():
                    tar.add(str(p), arcname=p.name)
                elif p.is_dir():
                    for child in p.rglob("*"):
                        if child.is_file():
                            tar.add(str(child), arcname=str(child.relative_to(p)))
        buf.seek(0)
        payload = buf.read()

        boto3, *_ = _require_aws_sdks()
        s3 = boto3.client("s3", region_name=cfg.region)
        prefix = cfg.s3_prefix or "ecs-sandbox"
        nonce = uuid.uuid4().hex[:12]
        key = f"{prefix}/{self._run_id}/{self._task_id}/upload-{nonce}.tar.gz"
        s3.put_object(Bucket=cfg.s3_bucket, Key=key, Body=payload)
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": cfg.s3_bucket, "Key": key},
            ExpiresIn=21600,
        )
        dl_cmd = (
            f"mkdir -p {shlex.quote(dest_dir)} && "
            f"TGZ=/tmp/_upload_$$.tar.gz && "
            f"( curl -sf -L --max-time 300 -o $TGZ {shlex.quote(url)} 2>/dev/null || "
            f"python3 -c 'import urllib.request as u,sys;u.urlretrieve(sys.argv[1],sys.argv[2])' "
            f"{shlex.quote(url)} $TGZ ) && "
            f"tar xzf $TGZ -C {shlex.quote(dest_dir)} && rm -f $TGZ && echo ok"
        )
        result = self._exec_client.exec(dl_cmd, timeout=360)  # type: ignore[union-attr]
        if "ok" not in result.stdout:
            raise RuntimeError(
                f"S3 upload extraction failed (rc={result.return_code}): {result.stderr or result.stdout}"
            )

    # Atexit cleanup --------------------------------------------------

    def _register_for_cleanup(self) -> None:
        global _atexit_registered
        with _cleanup_lock:
            _active_sandboxes[id(self)] = self
            if not _atexit_registered:
                atexit.register(_emergency_cleanup)
                _atexit_registered = True

    def _unregister_from_cleanup(self) -> None:
        with _cleanup_lock:
            _active_sandboxes.pop(id(self), None)
