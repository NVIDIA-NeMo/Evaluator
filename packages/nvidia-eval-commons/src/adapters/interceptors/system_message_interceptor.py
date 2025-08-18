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

"""System message interceptor with registry support."""

import json
from typing import final

from flask import Request
from nvidia_eval_commons.adapters.decorators import register_for_adapter
from nvidia_eval_commons.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    RequestInterceptor,
)
from pydantic import BaseModel, Field


@register_for_adapter(
    name="system_message",
    description="Adds system message to requests",
)
@final
class SystemMessageInterceptor(RequestInterceptor):
    """Adds system message to requests."""

    class Params(BaseModel):
        """Configuration parameters for system message interceptor."""

        system_message: str = Field(
            ..., description="System message to add to requests"
        )

    system_message: str

    def __init__(self, params: Params):
        """
        Initialize the system message interceptor.

        Args:
            params: Configuration parameters
        """
        self.system_message = params.system_message

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest:
        new_data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": self.system_message},
                    *json.loads(ar.r.get_data())["messages"],
                ],
                **{
                    k: v
                    for k, v in json.loads(ar.r.get_data()).items()
                    if k != "messages"
                },
            }
        )

        new_request = Request.from_values(
            path=ar.r.path,
            headers=dict(ar.r.headers),
            data=new_data,
            method=ar.r.method,
        )
        return AdapterRequest(
            r=new_request,
            rctx=ar.rctx,
        )
