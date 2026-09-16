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
"""ToolSandbox NEL entrypoint.

Patches the ToolSandbox agent/user factories to use NVIDIA-hosted models via
any OpenAI-compatible endpoint, then delegates to the standard tool_sandbox CLI.

Required environment variables:
  NVIDIA_BASE_URL    – OpenAI-compatible endpoint base URL
                       (e.g. https://integrate.api.nvidia.com/v1)
  NVIDIA_API_KEY     – API key for both agent and user models
  NVIDIA_AGENT_MODEL – Model name for the agent under evaluation

Optional:
  NVIDIA_USER_MODEL  – Model for user simulator
                       (default: meta/llama-3.1-70b-instruct)

CLI args (after patching) follow the standard tool_sandbox interface:
  --agent Gorilla --user GPT_4_o_2024_05_13 [--scenarios ...] [--test_mode]
"""
from __future__ import annotations

import os
import sys


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        raise RuntimeError(f"Required environment variable {name!r} is not set")
    return val


def _register_nvidia_roles() -> None:
    """Replace Gorilla agent and GPT-4o user with NVIDIA NIM-backed classes.

    We reuse existing RoleImplType enum keys so the CLI accepts
    ``--agent Gorilla --user GPT_4_o_2024_05_13`` without modification.
    """
    from openai import OpenAI
    from tool_sandbox.cli.utils import AGENT_TYPE_TO_FACTORY, RoleImplType, USER_TYPE_TO_FACTORY
    from tool_sandbox.roles.openai_api_agent import OpenAIAPIAgent
    from tool_sandbox.roles.openai_api_user import OpenAIAPIUser

    base_url = _require_env("NVIDIA_BASE_URL")
    api_key = _require_env("NVIDIA_API_KEY")
    agent_model = _require_env("NVIDIA_AGENT_MODEL")
    user_model = os.environ.get("NVIDIA_USER_MODEL", "meta/llama-3.1-70b-instruct")

    # OpenAIAPIAgent/User.__init__ reads OPENAI_API_KEY to create a temporary
    # client that NVIDIANIMAgent immediately replaces.  Set a placeholder so
    # the parent __init__ doesn't raise when the env var is absent.
    os.environ.setdefault("OPENAI_API_KEY", api_key or "not-used")

    def _client() -> OpenAI:
        return OpenAI(base_url=base_url, api_key=api_key)

    class NVIDIANIMAgent(OpenAIAPIAgent):
        model_name: str = agent_model

        def __init__(self) -> None:
            super().__init__()
            self.openai_client = _client()

    class NVIDIANIMUser(OpenAIAPIUser):
        model_name: str = user_model

        def __init__(self) -> None:
            super().__init__()
            self.openai_client = _client()

    AGENT_TYPE_TO_FACTORY[RoleImplType.Gorilla] = NVIDIANIMAgent
    USER_TYPE_TO_FACTORY[RoleImplType.GPT_4_o_2024_05_13] = NVIDIANIMUser


if __name__ == "__main__":
    _register_nvidia_roles()
    from tool_sandbox.cli import main

    sys.exit(main())
