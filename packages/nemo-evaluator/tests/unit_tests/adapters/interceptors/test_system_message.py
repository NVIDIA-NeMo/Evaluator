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

import json

from flask import Request
from nemo_evaluator.adapters.interceptors.system_message_interceptor import (
    SystemMessageInterceptor,
)
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterRequestContext,
)


def mock_context(tmpdir):
    return AdapterGlobalContext(output_dir=str(tmpdir), url="http://localhost")


def test_new_system_injected(tmpdir):
    # Test if the new system message is injected at the beginning

    system_message_interceptor = SystemMessageInterceptor(
        params=SystemMessageInterceptor.Params(system_message="detailed thinking on")
    )
    data = {
        "model": "gpt-4.1",
        "messages": [
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": "Are semicolons optional in JavaScript?"},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    request = Request.from_values(
        method="POST",
        headers={"Content-Type": "application/json"},
        data=json.dumps(data),
    )
    adapter_request = AdapterRequest(
        r=request,
        rctx=AdapterRequestContext(),
    )
    adapter_response = system_message_interceptor.intercept_request(
        adapter_request, mock_context(tmpdir)
    )
    json_output = adapter_response.r.get_json()
    assert json_output["messages"][0]["content"] == "detailed thinking on"
