# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Tuple

import yaml
from nemo_evaluator.adapters.adapter_config import AdapterConfig
from nemo_evaluator.adapters.server import (
    AdapterServer,
    AdapterServerProcess,
    wait_for_server,
)
from nemo_evaluator.api.api_dataclasses import (
    ApiEndpoint,
    Evaluation,
    EvaluationConfig,
    EvaluationTarget,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch only the Adapter Server using adapter_config from a YAML config."
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config that contains adapter_config",
    )
    parser.add_argument(
        "--host",
        default=None,
        help=(
            "Host for the adapter server (overrides ADAPTER_HOST). Default is 'localhost'."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=("Port for the adapter server (overrides ADAPTER_PORT). Fails if taken."),
    )
    parser.add_argument(
        "--upstream-url",
        default=None,
        help=(
            "Override upstream API URL. If not set, read target.api_endpoint.url from YAML."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for adapter outputs. If not set, derive from YAML or /tmp/nemo-adapter."
        ),
    )
    parser.add_argument(
        "--print-port",
        action="store_true",
        help="Print the effective adapter port to stdout after start",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for this launcher",
    )
    return parser.parse_args()


def load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(d: dict[str, Any], path: Tuple[str, ...], default=None):
    curr = d
    for key in path:
        if not isinstance(curr, dict) or key not in curr:
            return default
        curr = curr[key]
    return curr


def extract_adapter_config(config: dict[str, Any]) -> tuple[dict[str, Any] | None, str]:
    warnings: list[str] = []

    # Preferred location: target.api_endpoint.adapter_config
    direct = _deep_get(config, ("target", "api_endpoint", "adapter_config"))
    if isinstance(direct, dict):
        return direct, ""

    # Launcher-style global: evaluation.nemo_evaluator_config.config.target.api_endpoint.adapter_config
    global_eval = _deep_get(
        config,
        (
            "evaluation",
            "nemo_evaluator_config",
            "config",
            "target",
            "api_endpoint",
            "adapter_config",
        ),
    )
    if isinstance(global_eval, dict):
        return global_eval, ""

    # Fallback: first task-level adapter_config if present
    tasks = _deep_get(config, ("evaluation", "tasks"))
    if isinstance(tasks, list):
        for task in tasks:
            task_cfg = _deep_get(
                task,
                (
                    "nemo_evaluator_config",
                    "config",
                    "target",
                    "api_endpoint",
                    "adapter_config",
                ),
            )
            if isinstance(task_cfg, dict):
                warnings.append(
                    "Using task-level adapter_config (no global adapter_config found)."
                )
                return task_cfg, ("\n".join(warnings) if warnings else "")

    return None, ("\n".join(warnings) if warnings else "")


def resolve_upstream_url(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    url = _deep_get(config, ("target", "api_endpoint", "url"))
    if isinstance(url, str) and url:
        return url
    raise SystemExit(
        "Upstream URL is not provided. Use --upstream-url or set target.api_endpoint.url in YAML."
    )


def resolve_output_dir(config: dict[str, Any], override: str | None) -> str:
    if override:
        return override
    # Evaluator-style
    out = _deep_get(config, ("config", "output_dir"))
    if isinstance(out, str) and out:
        return out
    # Launcher-style
    out = _deep_get(config, ("execution", "output_dir"))
    if isinstance(out, str) and out:
        return out
    return "/tmp/nemo-adapter"


def build_typed_adapter_config(
    adapter_cfg_dict: dict[str, Any], output_dir: str
) -> AdapterConfig:
    run_config = {
        "target": {"api_endpoint": {"adapter_config": adapter_cfg_dict}},
        "config": {"output_dir": output_dir},
    }
    return AdapterConfig.get_validated_config(run_config)


def precheck_enabled(adapter_cfg: AdapterConfig) -> None:
    enabled_interceptors = [ic for ic in adapter_cfg.interceptors if ic.enabled]
    enabled_hooks = [h for h in adapter_cfg.post_eval_hooks if h.enabled]
    if not enabled_interceptors and not enabled_hooks:
        # Produce a message similar to server validation expectations
        names_interceptors = [ic.name for ic in adapter_cfg.interceptors]
        names_hooks = [h.name for h in adapter_cfg.post_eval_hooks]
        msg = (
            "Adapter server cannot start: No enabled interceptors or post-eval hooks found. "
            f"Configured interceptors: {names_interceptors}; "
            f"Configured post-eval hooks: {names_hooks}"
        )
        raise SystemExit(msg)


def build_evaluation(
    upstream_url: str, output_dir: str, adapter_cfg: AdapterConfig
) -> Evaluation:
    eval_cfg = EvaluationConfig(output_dir=output_dir)
    endpoint = ApiEndpoint(url=upstream_url, adapter_config=adapter_cfg)
    target = EvaluationTarget(api_endpoint=endpoint)
    return Evaluation(
        command="",
        framework_name="",
        pkg_name="",
        config=eval_cfg,
        target=target,
    )


def main() -> None:
    args = parse_args()

    config = load_yaml(args.config)

    adapter_cfg_dict, warn = extract_adapter_config(config)
    if warn:
        print(warn, file=sys.stderr)
    if not isinstance(adapter_cfg_dict, dict):
        raise SystemExit(
            "No adapter_config found in YAML. Expected under target.api_endpoint or evaluation.nemo_evaluator_config.config.target.api_endpoint."
        )

    upstream_url = resolve_upstream_url(config, args.upstream_url)
    output_dir = resolve_output_dir(config, args.output_dir)

    try:
        typed_adapter_cfg = build_typed_adapter_config(adapter_cfg_dict, output_dir)
    except Exception as e:
        raise SystemExit(f"Invalid adapter configuration: {e}") from e

    # Pre-validate enabled components for clearer early error message
    precheck_enabled(typed_adapter_cfg)

    evaluation = build_evaluation(upstream_url, output_dir, typed_adapter_cfg)

    # Respect host/port overrides via environment variables used by AdapterServerProcess
    if args.host:
        os.environ["ADAPTER_HOST"] = args.host
    if args.port is not None:
        os.environ["ADAPTER_PORT"] = str(args.port)

    # Start server process and block until interrupted
    try:
        with AdapterServerProcess(evaluation) as proc:
            if proc is None:
                raise SystemExit(
                    "Adapter server was not started because no enabled interceptors or post-eval hooks were configured."
                )

            host = os.environ.get("ADAPTER_HOST", AdapterServer.DEFAULT_ADAPTER_HOST)
            port = proc.port

            # Wait (redundantly) until server is ready before announcing
            wait_for_server(host, port)

            if args.print_port:
                # Only print port number to stdout for easy scripting
                print(port)

            print(
                f"Adapter server running on http://{host}:{port} (upstream: {proc.original_url})",
                file=sys.stderr,
            )

            # Block until user stops the process
            try:
                while True:
                    time.sleep(1.0)
            except KeyboardInterrupt:
                # Exit context manager to trigger graceful shutdown and post-eval hooks
                pass

    except KeyboardInterrupt:
        # Graceful termination path
        pass
    except SystemExit:
        raise
    except Exception as e:
        raise SystemExit(f"Failed to start adapter server: {e}") from e


if __name__ == "__main__":
    main()
