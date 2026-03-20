"""LiteLLM proxy lifecycle — start / health-check / stop.

The proxy runs as a child process (``uvicorn`` serving the LiteLLM app)
so that LiteLLM's global-state mutations don't affect the evaluator.
A temporary config YAML is generated to route traffic to the upstream
model endpoint and register interceptors.
"""
from __future__ import annotations

import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 0.5
_HEALTH_TIMEOUT = 30.0


def _require_litellm() -> None:
    try:
        import litellm  # noqa: F401
    except ImportError:
        raise ImportError(
            "LiteLLM proxy requires the litellm package. "
            "Install with: pip install nemo-evaluator[proxy]"
        ) from None
    try:
        from litellm.proxy.proxy_server import app  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            f"LiteLLM is installed but proxy extras are missing ({exc}). "
            f"Install with: pip install 'litellm[proxy]'"
        ) from None


@dataclass
class ProxyHandle:
    """Handle to a running LiteLLM proxy subprocess."""

    url: str
    port: int
    log_file: Path
    _process: subprocess.Popen[bytes] = field(repr=False)
    _config_path: Path = field(repr=False)

    def stop(self) -> None:
        if self._process.poll() is not None:
            return
        logger.info("Stopping LiteLLM proxy (pid=%d)", self._process.pid)
        self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Proxy did not exit gracefully — killing")
            self._process.kill()
            self._process.wait(timeout=5)
        self._dump_tail()
        try:
            self._config_path.unlink(missing_ok=True)
        except OSError:
            pass

    def _dump_tail(self, n_lines: int = 40) -> None:
        """Log the last *n_lines* of the proxy log file for post-mortem."""
        try:
            text = self.log_file.read_text(errors="replace")
        except OSError:
            return
        lines = text.strip().splitlines()
        if not lines:
            return
        tail = lines[-n_lines:]
        logger.info(
            "LiteLLM proxy log tail (%d/%d lines, full log: %s):\n%s",
            len(tail), len(lines), self.log_file, "\n".join(tail),
        )


def _strip_chat_completions(url: str) -> str:
    """Strip trailing ``/chat/completions`` so LiteLLM doesn't double it."""
    for suffix in ("/chat/completions", "/chat/completions/"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def _build_config(
    model_url: str,
    model_id: str,
    api_key: str | None,
    callbacks: list[Any],
    *,
    verbose: bool = False,
) -> dict[str, Any]:
    """Build a LiteLLM proxy config dict."""
    litellm_params: dict[str, Any] = {
        "model": f"openai/{model_id}",
        "api_base": _strip_chat_completions(model_url),
    }
    if api_key:
        litellm_params["api_key"] = api_key

    cfg: dict[str, Any] = {
        "model_list": [
            {
                "model_name": model_id,
                "litellm_params": litellm_params,
            },
        ],
    }

    litellm_settings: dict[str, Any] = {}
    string_callbacks = [c for c in callbacks if isinstance(c, str)]
    if string_callbacks:
        litellm_settings["callbacks"] = string_callbacks
    if verbose:
        litellm_settings["set_verbose"] = True
    if litellm_settings:
        cfg["litellm_settings"] = litellm_settings

    return cfg


def _write_temp_config(cfg: dict[str, Any]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="nel_litellm_")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(cfg, f)
    return Path(path)


def _port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def _find_free_port(preferred: int) -> int:
    """Return *preferred* if available, otherwise pick a random free port."""
    if _port_is_free(preferred):
        return preferred
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_healthy(port: int, timeout: float = _HEALTH_TIMEOUT) -> None:
    """Poll the proxy health endpoint until it responds or times out."""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health/liveliness"
    deadline = time.monotonic() + timeout
    last_err: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2.0) as r:
                if r.status == 200:
                    return
        except (urllib.error.URLError, OSError) as exc:
            last_err = exc
        time.sleep(_HEALTH_POLL_INTERVAL)
    raise TimeoutError(
        f"LiteLLM proxy on port {port} did not become healthy "
        f"within {timeout}s: {last_err}"
    )


def start_proxy(
    model_url: str,
    model_id: str,
    api_key: str | None,
    *,
    port: int = 4000,
    interceptors: list[str | dict[str, Any]] | None = None,
    verbose: bool = False,
) -> ProxyHandle:
    """Start a local LiteLLM proxy and return a handle.

    The proxy forwards ``/v1/*`` requests to *model_url* using *api_key*,
    running any configured interceptors on each request/response.
    """
    _require_litellm()

    actual_port = _find_free_port(port)
    if actual_port != port:
        logger.info("Port %d in use — using %d instead", port, actual_port)
    port = actual_port

    from nemo_evaluator.interceptors import resolve_interceptors

    resolved = resolve_interceptors(interceptors or [])

    custom_objects = [cb for cb in resolved if not isinstance(cb, str)]
    all_callbacks = [cb for cb in resolved if isinstance(cb, str)]

    cfg = _build_config(model_url, model_id, api_key, all_callbacks, verbose=verbose)
    config_path = _write_temp_config(cfg)

    logger.info(
        "Starting LiteLLM proxy on port %d → %s (interceptors: %s)",
        port, model_url, interceptors or [],
    )

    env = {**os.environ, "CONFIG_FILE_PATH": str(config_path)}

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="nel_litellm_")
    log_file = Path(log_path)
    log_fh = os.fdopen(log_fd, "wb")

    if verbose:
        out_target = sys.stderr
        err_target: int | Any = subprocess.STDOUT
    else:
        out_target = log_fh
        err_target = subprocess.STDOUT

    if custom_objects:
        _register_custom_callbacks_script = (
            "import litellm; "
            "from nemo_evaluator.interceptors import resolve_interceptors; "
            f"cbs = resolve_interceptors({interceptors!r}); "
            "litellm.callbacks.extend([c for c in cbs if not isinstance(c, str)]); "
            "from litellm.proxy.proxy_server import app; "
            "import uvicorn; "
            f"uvicorn.run(app, host='127.0.0.1', port={port}, log_level='warning')"
        )
        proc = subprocess.Popen(
            [sys.executable, "-c", _register_custom_callbacks_script],
            env=env,
            stdout=out_target,
            stderr=err_target,
        )
    else:
        proc = subprocess.Popen(
            [
                sys.executable, "-m", "uvicorn",
                "litellm.proxy.proxy_server:app",
                "--host", "127.0.0.1",
                "--port", str(port),
                "--log-level", "warning",
            ],
            env=env,
            stdout=out_target,
            stderr=err_target,
        )
    log_fh.close()

    try:
        _wait_for_healthy(port)
    except Exception:
        try:
            stderr_output = log_file.read_text(errors="replace")
        except OSError:
            stderr_output = ""
        proc.kill()
        proc.wait()
        config_path.unlink(missing_ok=True)
        if stderr_output:
            logger.error("LiteLLM proxy log:\n%s", stderr_output[-4000:])
        raise

    logger.info(
        "LiteLLM proxy ready (pid=%d, port=%d, log=%s)",
        proc.pid, port, log_file,
    )
    return ProxyHandle(
        url=f"http://127.0.0.1:{port}/v1",
        port=port,
        log_file=log_file,
        _process=proc,
        _config_path=config_path,
    )
