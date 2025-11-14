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

import hashlib
import os
import subprocess
import tempfile
import time
from typing import Any, Literal, TypeVar

import requests
import yaml

from nemo_evaluator.logging import get_logger

__all__ = []

logger = get_logger(__name__)


class MisconfigurationError(Exception):
    pass


KeyType = TypeVar("KeyType")


def deep_update(
    mapping: dict[KeyType, Any],
    *updating_mappings: dict[KeyType, Any],
    skip_nones: bool = False,
) -> dict[KeyType, Any]:
    """Deep update a mapping with other mappings.

    If `skip_nones` is True, then the values that are None in the updating mappings are
    not updated.
    """
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(
                    updated_mapping[k], v, skip_nones=skip_nones
                )
            else:
                if skip_nones and v is None:
                    continue
                updated_mapping[k] = v
    return updated_mapping


def dotlist_to_dict(dotlist: list[str]) -> dict:
    """Resolve dot-list style key-value pairs with YAML.

    Helper for overriding configuration values using command-line arguments in dot-list style.
    """
    dotlist_dict = {}
    for override in dotlist:
        parts = override.strip().split("=", 1)
        if len(parts) == 2:
            key = parts[0].strip()
            raw_value = parts[1].strip()

            # If the value starts with a quote but doesn't end with the same quote,
            # it means we have a malformed string. In this case, we'll treat it as a raw string.
            if (raw_value.startswith('"') and not raw_value.endswith('"')) or (
                raw_value.startswith("'") and not raw_value.endswith("'")
            ):
                value = raw_value
            else:
                try:
                    value = yaml.safe_load(raw_value)
                except yaml.YAMLError:
                    # If YAML parsing fails, treat it as a raw string
                    value = raw_value

            keys = key.split(".")
            temp = dotlist_dict
            for k in keys[:-1]:
                temp = temp.setdefault(k, {})
            temp[keys[-1]] = value
    return dotlist_dict


def run_command(command, cwd=None, verbose=False, propagate_errors=False):
    if verbose:
        logger.info(f"Running command: {command}")
        if cwd:
            print(f"Current working directory set to: {cwd}")

    with tempfile.TemporaryDirectory() as tmpdirname:
        if verbose:
            logger.info(f"Temporary directory created at: {tmpdirname}")

        file = os.path.join(
            tmpdirname, hashlib.sha1(command.encode("utf-8")).hexdigest() + ".sh"
        )
        if verbose:
            logger.info(f"Script file created: {file}")

        with open(file, "w") as f:
            f.write(command)
            f.flush()
            if verbose:
                logger.info("Command written to script file.")

        master, slave = os.openpty()
        process = subprocess.Popen(
            f"bash {file}",
            stdout=slave,
            stderr=slave,
            stdin=subprocess.PIPE,
            cwd=cwd,
            shell=True,
            executable="/bin/bash",
        )

        if verbose:
            logger.info("Subprocess started.")

        os.close(slave)

        if propagate_errors:
            stderr_output = []

        while True:
            try:
                output = os.read(master, 1024)
                if not output:
                    break
                decoded_output = output.decode(errors="ignore")
                print(decoded_output, end="", flush=True)

                if propagate_errors:
                    stderr_output.append(decoded_output)

            except OSError as e:
                if e.errno == 5:  # Input/output error is expected at the end of output
                    break
                raise

        if verbose:
            logger.info("Output reading completed.")

        rc = process.wait()

        if verbose:
            logger.info(f"Subprocess finished with return code: {rc}")

        # New error propagation logic
        if rc != 0 and propagate_errors:
            error_content = (
                "".join(stderr_output) if stderr_output else "No error details captured"
            )
            raise RuntimeError(
                f"Evaluation failed! Please consult the logs below:\n{error_content}"
            )

        return rc


def check_health(
    health_url: str, max_retries: int = 600, retry_interval: int = 2
) -> bool:
    """
    Check the health of the server.
    """
    for _ in range(max_retries):
        try:
            response = requests.get(health_url)
            if response.status_code == 200:
                return True
            logger.info(f"Server replied with status code: {response.status_code}")
            time.sleep(retry_interval)
        except requests.exceptions.RequestException:
            logger.info("Server is not ready")
            time.sleep(retry_interval)
    return False


def check_endpoint(
    endpoint_url: str,
    endpoint_type: Literal["completions", "chat"],
    model_name: str,
    max_retries: int = 600,
    retry_interval: int = 2,
) -> bool:
    """Checks if the OpenAI-compatible endpoint is alive by sending a simple prompt.

    Args:
        endpoint_url (str): Full endpoint URL. For most servers that means either ``/v1/chat/completions`` or ``/completions`` must be provided
        endpoint_type (Literal[completions, chat]): indicates if the model is instruction-tuned (chat) or a base model (completions). Used to constuct a proper payload structure.
        model_name (str): model name that is linked to payload. Might be required by some endpoint.
        max_retries (int, optional): How many attempt before returning false. Defaults to 600.
        retry_interval (int, optional): How many seconds to wait between attempts. Defaults to 2.

    Raises:
        ValueError: if endpoint_type was not one of "completions", "chat"

    Returns:
        bool: whether the endpoint is alive
    """
    payload = {"model": model_name, "max_tokens": 1}
    if endpoint_type == "completions":
        payload["prompt"] = "hello, my name is"
    elif endpoint_type == "chat":
        payload["messages"] = [{"role": "user", "content": "hello, what is your name?"}]
    else:
        raise ValueError(f"Invalid endpoint type: {endpoint_type}")

    for _ in range(max_retries):
        try:
            response = requests.post(endpoint_url, json=payload)
            if response.status_code == 200:
                return True
            logger.info(f"Server replied with status code: {response.status_code}")
            time.sleep(retry_interval)
        except requests.exceptions.RequestException:
            logger.info("Server is not ready")
            time.sleep(retry_interval)
    return False
