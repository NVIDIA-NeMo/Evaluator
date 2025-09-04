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

"""Reasoning interceptor that strips reasoning from responses and tracks reasoning information."""

import json
import re
from typing import final

from pydantic import Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterResponse,
    ResponseInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@register_for_adapter(
    name="reasoning",
    description="Strips reasoning from responses and tracks reasoning information",
)
@final
class ResponseReasoningInterceptor(ResponseInterceptor):
    """Adds reasoning information to responses and tracks reasoning metrics."""

    class Params(BaseLoggingParams):
        """Configuration parameters for reasoning interceptor."""

        end_reasoning_token: str = Field(
            default="</think>",
            description="Token that marks the end of reasoning section, not used if reasoning_content is provided",
        )
        start_reasoning_token: str | None = Field(
            default="<think>",
            description="Token that marks the start of reasoning section, used for tracking if reasoning has started",
        )
        add_reasoning: bool = Field(
            default=True, description="Whether to add reasoning information"
        )
        enable_reasoning_tracking: bool = Field(
            default=True, description="Enable reasoning tracking and logging"
        )
        include_if_not_finished: bool = Field(
            default=True,
            description="Include reasoning content if reasoning is not finished (end token not found)",
        )

    end_reasoning_token: str
    start_reasoning_token: str | None
    add_reasoning: bool
    enable_reasoning_tracking: bool
    include_if_not_finished: bool

    def __init__(self, params: Params):
        """
        Initialize the reasoning interceptor.

        Args:
            params: Configuration parameters
        """
        self.end_reasoning_token = params.end_reasoning_token
        self.start_reasoning_token = params.start_reasoning_token
        self.add_reasoning = params.add_reasoning
        self.enable_reasoning_tracking = params.enable_reasoning_tracking
        self.include_if_not_finished = params.include_if_not_finished

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Reasoning interceptor initialized",
            end_reasoning_token=self.end_reasoning_token,
            start_reasoning_token=self.start_reasoning_token,
            add_reasoning=self.add_reasoning,
            enable_reasoning_tracking=self.enable_reasoning_tracking,
            include_if_not_finished=self.include_if_not_finished,
        )

    def _process_reasoning_message(self, msg: dict) -> tuple[dict, dict]:
        """
        Process reasoning in the message and return modified message with reasoning info.

        Args:
            msg: The message object containing content and potentially reasoning_content

        Returns:
            tuple: (modified_message, reasoning_info) where reasoning_info has keys:
                   reasoning_words, original_content_words, updated_content_words, reasoning_finished, reasoning_started
        """
        modified_msg = msg.copy()
        content = msg.get("content", "")

        # Check if reasoning_content exists in the message and is not empty
        if (
            "reasoning_content" in msg
            and msg["reasoning_content"]
            and msg["reasoning_content"].strip()
        ):
            reasoning_content = msg["reasoning_content"]
            updated_message_content = content
            reasoning_started = True
            if content.strip() == "":
                reasoning_finished = False
            else:
                reasoning_finished = True
        else:
            reasoning_finished = False
            if self.start_reasoning_token is not None:
                reasoning_started = self.start_reasoning_token in content
            else:
                reasoning_started = "unknown"
            if self.end_reasoning_token in content:
                reasoning_finished = True
                reasoning_started = True

            # Split content using reasoning token
            if reasoning_finished:
                cleaned_content = self._strip_reasoning(content)
                reasoning_content = content[: content.find(self.end_reasoning_token)]
                updated_message_content = cleaned_content
            else:
                if reasoning_started == "unknown":
                    reasoning_content = "unknown"
                elif reasoning_started:
                    reasoning_content = content
                else:
                    reasoning_content = ""
                if not self.include_if_not_finished:
                    updated_message_content = ""
                else:
                    updated_message_content = content

        # Assign the updated message content
        modified_msg["content"] = updated_message_content

        # Calculate lengths and reasoning status
        if reasoning_content and reasoning_content != "unknown":
            reasoning_words = len(reasoning_content.split())
        elif reasoning_started == "unknown":
            reasoning_words = "unknown"
        else:
            reasoning_words = 0

        reasoning_info = {
            "reasoning_words": reasoning_words,
            "original_content_words": (len(content.split()) if content else 0),
            "updated_content_words": (
                len(modified_msg.get("content", "").split())
                if modified_msg.get("content")
                else 0
            ),
            "reasoning_finished": reasoning_finished,
            "reasoning_started": reasoning_started,
        }

        return modified_msg, reasoning_info

    def _strip_reasoning(self, text: str) -> str:
        """Remove everything between start and end reasoning tokens."""
        # Remove everything between start and end reasoning tokens
        # Also handle cases where only end token is present
        cleaned_content = re.sub(
            r".*?" + re.escape(self.end_reasoning_token),
            "",
            text,
            flags=re.DOTALL,
        ).strip("\n")
        return cleaned_content

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Remove reasoning tokens from assistant message content in the response and track reasoning info."""
        if not self.add_reasoning:
            self.logger.debug("Reasoning processing disabled, returning response as-is")
            return resp

        try:
            response_data = resp.r.json()

            if isinstance(response_data, dict) and "choices" in response_data:
                self.logger.debug(
                    "Processing response with choices",
                    choices_count=len(response_data["choices"]),
                )

                for choice in response_data["choices"]:
                    msg = choice.get("message")
                    if (
                        msg
                        and msg.get("role") == "assistant"
                        and isinstance(msg.get("content"), str)
                    ):
                        # Get modified message and reasoning information
                        modified_msg, reasoning_info = self._process_reasoning_message(
                            msg
                        )

                        # Log reasoning information if tracking is enabled
                        if self.enable_reasoning_tracking:
                            self.logger.info(
                                "Reasoning tracking information", **reasoning_info
                            )

                        # Update the message with the modified content
                        msg.update(modified_msg)

                        self.logger.debug(
                            "Message processed",
                            role=msg.get("role"),
                            original_content_length=reasoning_info[
                                "original_content_words"
                            ],
                            updated_content_length=reasoning_info[
                                "updated_content_words"
                            ],
                            reasoning_words=reasoning_info["reasoning_words"],
                        )
                # Optionally handle list responses if needed

            resp.r._content = json.dumps(response_data).encode()

            self.logger.info(
                "Response reasoning processing completed",
                response_keys=(
                    list(response_data.keys())
                    if isinstance(response_data, dict)
                    else "unknown"
                ),
            )

        except Exception as e:
            # If we can't parse the response, just return it as is
            self.logger.error("Failed to process response reasoning", error=str(e))
            pass

        return resp
