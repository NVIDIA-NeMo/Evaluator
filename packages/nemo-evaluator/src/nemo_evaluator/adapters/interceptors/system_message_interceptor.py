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
from pydantic import Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    RequestInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@register_for_adapter(
    name="system_message",
    description="Adds or replaces system message in requests.",
)
@final
class SystemMessageInterceptor(RequestInterceptor):
    """Adds or replaces system message in requests."""

    class Params(BaseLoggingParams):
        """Configuration parameters for system message interceptor."""

        system_message: str = Field(
            ..., description="System message to add to requests"
        )

    def __init__(self, params: Params):
        """
        Initialize the system message interceptor.

        Args:
            params: Configuration parameters
        """
        self.system_message = params.system_message

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "System message interceptor initialized",
            system_message_preview=(
                self.system_message[:100] + "..."
                if len(self.system_message) > 100
                else self.system_message
            ),
        )

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest:
        original_data = json.loads(ar.r.get_data())

        self.logger.debug(
            "Processing request for system message addition",
            original_messages_count=len(original_data.get("messages", [])),
            has_system_message=any(
                msg.get("role") == "system" for msg in original_data.get("messages", [])
            ),
        )

        new_data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": self.system_message},
                    *[
                        msg
                        for msg in json.loads(ar.r.get_data())["messages"]
                        if msg["role"] != "system"
                    ],
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

        self.logger.debug(
            "System message added to request",
            original_messages_count=len(original_data.get("messages", [])),
            new_messages_count=len(original_data.get("messages", [])) + 1,
            system_message_length=len(self.system_message),
        )

        return AdapterRequest(
            r=new_request,
            rctx=ar.rctx,
        )
