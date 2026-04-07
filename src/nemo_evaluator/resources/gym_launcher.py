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
#!/usr/bin/env python3
"""Launch a Gym resource server with minimal config.

Gym servers use Hydra/OmegaConf for configuration, not CLI flags.
This launcher sets up the NEMO_GYM_CONFIG_DICT env var and calls
``run_webserver()`` on the requested server class.

Usage:
    python -m nemo_evaluator.resources.gym_launcher \
        resources_servers.spider2_lite.app:Spider2LiteResourcesServer \
        --port 9100

    # With extra config keys:
    python -m nemo_evaluator.resources.gym_launcher \
        resources_servers.spider2_lite.app:Spider2LiteResourcesServer \
        --port 9100 \
        -c spider2_lite_dir=resources_servers/spider2_lite/.spider2_lite

    # Auto-download missing datasets before starting:
    python -m nemo_evaluator.resources.gym_launcher \
        resources_servers.spider2_lite.app:Spider2LiteResourcesServer \
        --port 9100 --prepare-data \
        -c spider2_lite_dir=resources_servers/spider2_lite/.spider2_lite
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path


def _download_from_gitlab(gl_id: dict, output_path: str) -> None:
    """Download a dataset from GitLab MLflow without importing nemo_gym.

    Requires MLFLOW_TRACKING_URI and MLFLOW_TRACKING_TOKEN env vars.
    """
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    tracking_token = os.environ.get("MLFLOW_TRACKING_TOKEN")
    if not tracking_uri or not tracking_token:
        raise EnvironmentError(
            "MLFLOW_TRACKING_URI and MLFLOW_TRACKING_TOKEN must be set to download datasets from GitLab MLflow."
        )

    from mlflow import MlflowClient
    from mlflow.artifacts import get_artifact_repository
    import requests

    os.environ["MLFLOW_TRACKING_TOKEN"] = tracking_token
    client = MlflowClient(tracking_uri=tracking_uri)

    model_version = client.get_model_version(gl_id["dataset_name"], gl_id["version"])
    run_id = model_version.run_id
    repo = get_artifact_repository(
        artifact_uri=f"runs:/{run_id}",
        tracking_uri=tracking_uri,
    )
    artifact_uri = repo.repo.artifact_uri
    download_link = f"{artifact_uri.rstrip('/')}/{gl_id['artifact_fpath'].lstrip('/')}"

    response = requests.get(
        download_link,
        headers={"Authorization": f"Bearer {tracking_token}"},
    )
    response.raise_for_status()
    with open(output_path, "w") as f:
        f.write(response.content.decode())


def _download_from_hf(hf_id: dict, output_path: str) -> None:
    """Download a dataset from HuggingFace Hub without importing nemo_gym."""
    from huggingface_hub import hf_hub_download
    import shutil

    token = os.environ.get("HF_TOKEN")
    artifact = hf_id.get("artifact_fpath")
    if artifact:
        downloaded = hf_hub_download(
            repo_id=hf_id["repo_id"],
            filename=artifact,
            repo_type="dataset",
            token=token,
        )
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(downloaded, output_path)
    else:
        from datasets import load_dataset

        ds = load_dataset(hf_id["repo_id"], split=hf_id.get("split", "validation"), token=token)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        ds.to_json(output_path)


def _prepare_datasets(config_path: str) -> None:
    """Read a Gym resource-server YAML config and download any missing datasets.

    Downloads directly via mlflow/huggingface_hub to avoid importing nemo_gym
    (which triggers Hydra CLI parsing).

    Best-effort: failures are logged as warnings but never prevent the server
    from starting.
    """
    try:
        import yaml
    except ImportError:
        print("[gym_launcher] WARNING: PyYAML not installed, skipping data prep")
        return

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    datasets: list[dict] = []
    for top_value in raw.values():
        if not isinstance(top_value, dict):
            continue
        for section in ("responses_api_agents", "resources_servers"):
            for srv in (top_value.get(section) or {}).values():
                if isinstance(srv, dict):
                    datasets.extend(srv.get("datasets") or [])

    for ds in datasets:
        fpath = ds.get("jsonl_fpath")
        if not fpath or Path(fpath).exists():
            continue

        print(f"[gym_launcher] Dataset missing: {fpath}")
        Path(fpath).parent.mkdir(parents=True, exist_ok=True)

        hf_id = ds.get("huggingface_identifier")
        gl_id = ds.get("gitlab_identifier")

        try:
            if hf_id:
                print(f"[gym_launcher] Downloading from HuggingFace: {hf_id['repo_id']}")
                _download_from_hf(hf_id, fpath)
            elif gl_id:
                print(f"[gym_launcher] Downloading from GitLab MLflow: {gl_id['dataset_name']} v{gl_id['version']}")
                _download_from_gitlab(gl_id, fpath)
            else:
                print(f"[gym_launcher] WARNING: No download identifier for {fpath}")
                continue
        except Exception as exc:
            print(f"[gym_launcher] WARNING: Could not download {fpath}: {exc}")
            print(
                "[gym_launcher]   Provide the dataset manually or set "
                "MLFLOW_TRACKING_URI / MLFLOW_TRACKING_TOKEN / HF_TOKEN."
            )
            continue

        if Path(fpath).exists():
            print(f"[gym_launcher] Downloaded: {fpath}")
        else:
            print(f"[gym_launcher] WARNING: File not found after download: {fpath}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch a Gym resource server")
    parser.add_argument(
        "server",
        help="module:ClassName (e.g. resources_servers.spider2_lite.app:Spider2LiteResourcesServer)",
    )
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--head-port", type=int, default=None, help="Head server port (default: port - 1)")
    parser.add_argument("--config", "-c", action="append", default=[], help="Extra config key=value pairs")
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Download missing datasets before starting",
    )
    parser.add_argument(
        "--gym-config",
        default=None,
        help="Path to the Gym resource server YAML config (for --prepare-data)",
    )
    args = parser.parse_args()

    # Hydra (pulled in transitively by nemo_gym) re-parses sys.argv on import.
    # We've captured everything we need via argparse; clear argv now.
    sys.argv = [sys.argv[0]]

    # --- Data preparation (before any nemo_gym imports) ---
    if args.prepare_data:
        gym_config = args.gym_config
        if not gym_config:
            short_name = args.server.rsplit(":", 1)[0].rsplit(".", 1)[0].rsplit(".", 1)[-1]
            gym_config = f"resources_servers/{short_name}/configs/{short_name}.yaml"
            print(f"[gym_launcher] --gym-config not set, guessing: {gym_config}")
        if Path(gym_config).exists():
            _prepare_datasets(gym_config)
        else:
            print(f"[gym_launcher] WARNING: Config not found at {gym_config}, skipping data prep")

    # --- Build Gym config and set env vars ---
    module_path, class_name = args.server.rsplit(":", 1)
    server_name = class_name.lower().replace("resourcesserver", "_resources_server")

    head_port = args.head_port or (args.port - 1)

    server_config: dict = {
        "entrypoint": "app.py",
        "host": args.host,
        "port": args.port,
        "name": server_name,
    }
    for kv in args.config:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except (json.JSONDecodeError, ValueError):
            pass
        server_config[k] = v

    top_key = server_name
    server_type = "resources_servers" if "resources" in module_path else "responses_api_agents"
    short_name = module_path.rsplit(".", 1)[0].rsplit(".", 1)[-1]

    config_dict = {
        "dry_run": False,
        "head_server": {"host": args.host, "port": head_port},
        top_key: {server_type: {short_name: server_config}},
    }

    os.environ["NEMO_GYM_CONFIG_DICT"] = json.dumps(config_dict)
    os.environ["NEMO_GYM_CONFIG_PATH"] = top_key

    # --- Now safe to import nemo_gym (env var set → Hydra bypassed) ---
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name)
    cls.run_webserver()


if __name__ == "__main__":
    main()
