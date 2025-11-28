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

        strategy: str = Field(
            description="Strategy to use for system message addition. "
            "Options: 'replace' (default) - replaces existing system message, "
            "'append' - appends a system message to existing message"
            "'prepend' - prepends system message to existing message",
            default="prepend",
        )

    def __init__(self, params: Params):
        """
        Initialize the system message interceptor.

        Args:
            params: Configuration parameters
        """
        self.system_message = params.system_message
        self.strategy = params.strategy

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        # API change warning for version 25.11
        if self.strategy == "prepend":
            self.logger.warning(
                "API Change Notice: As of nemo-evaluator version 25.11, the default behavior "
                "of the system_message interceptor has changed from 'replace' to 'prepend'. "
                "The interceptor now prepends the configured system message to any existing "
                "system message instead of replacing it. To restore the previous behavior, "
                "explicitly set 'strategy: replace' in your interceptor configuration."
            )

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

        # find the existing system message and save it to the instance variable
        # join the multiple system messages into a single string if there are multiple
        existing_system_message = [
            msg for msg in original_data["messages"] if msg["role"] == "system"
        ]
        existing_system_message = (
            "\n".join([msg.get("content") for msg in existing_system_message])
            if len(existing_system_message) > 0
            else None
        )

        new_system_message = None

        if self.strategy == "replace":
            new_system_message = self.system_message
        elif self.strategy == "append":
            if existing_system_message:
                new_system_message = (
                    existing_system_message + "\n" + self.system_message
                )
            else:
                new_system_message = self.system_message
        elif self.strategy == "prepend":
            if existing_system_message:
                new_system_message = (
                    self.system_message + "\n" + existing_system_message
                )
            else:
                new_system_message = self.system_message

        new_data = json.dumps(
            {
                "messages": [
                    {"role": "system", "content": new_system_message},
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
