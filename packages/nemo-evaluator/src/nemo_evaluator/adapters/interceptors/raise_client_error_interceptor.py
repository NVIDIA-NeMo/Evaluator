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

from typing import List, Optional, final

import requests
from pydantic import BaseModel, Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterResponse,
    ResponseInterceptor,
)


@register_for_adapter(
    name="raise_client_errors",
    description="Whether to raise exception on client errors with response status code 4xx instead of continuing (excludes 408 and 429)",
)
@final
class RaiseClientErrorInterceptor(ResponseInterceptor):
    """Adapter for handling non-retryable client error to raise an exception instead of continuing the benchmark."""

    class Params(BaseModel):
        """No configuration parameters supported for raise client error interceptor."""

        exclude_status_codes: Optional[List[int]] = Field(
            default=[408, 429],
            description="Status codes to exclude from raising client errors when present in status_code_range.",
        )
        status_codes: Optional[List[int]] | None = Field(
            default=None, description="List of status codes to raise exception."
        )
        status_code_range_start: Optional[int] | None = Field(
            default=400,
            description="Start range of status codes to raise exception. Use with status_code_range_end to define an inclusive range e.g. [400, 499].",
        )
        status_code_range_end: Optional[int] | None = Field(
            default=499,
            description="End range of status codes to raise exception. Use with status_code_range_start to define an inclusive range e.g. [400, 499].",
        )

    exclude_status_codes: List[int] | None
    status_codes: List[int] | None
    status_code_range_start: int | None
    status_code_range_end: int | None

    def __init__(self, params: Params):
        """
        Initialize the raise client error interceptor.

        Args:
            params: Configuration parameters
        """
        if params.exclude_status_codes and params.status_codes:
            overlap_status_codes = set(params.exclude_status_codes).intersection(
                set(params.status_codes)
            )
            if overlap_status_codes:
                raise ValueError(
                    f"status code(s) cannot be listed in both status_codes and exclude_status_codes: {overlap_status_codes}"
                )

        self.exclude_status_codes = params.exclude_status_codes
        self.status_codes = params.status_codes

        if params.status_code_range_start and params.status_code_range_end:
            if params.status_code_range_start >= params.status_code_range_end:
                raise ValueError(
                    f"Status code start and end is not a valid range: [{params.status_code_range_start}, {params.status_code_range_end}]"
                )
        self.status_code_range_start = params.status_code_range_start
        self.status_code_range_end = params.status_code_range_end

    def _format_exception(self, response: requests.Response) -> Exception:
        return Exception(
            f"Client error detected with raise_client_errors enabled: {response.status_code} {response.text}"
        )

    def _handle_client_error(self, response: requests.Response) -> requests.Response:
        """
        Raises exception on client errors and not continue

        Args:
            response: The API response object from requests

        Returns:
            Response
        """
        status_code = response.status_code
        if self.status_codes and status_code in self.status_codes:
            raise self._format_exception(response)

        if self.exclude_status_codes and status_code in self.exclude_status_codes:
            return response

        # Check range if defined
        if self.status_code_range_start and self.status_code_range_end:
            if (
                self.status_code_range_start
                <= response.status_code
                <= self.status_code_range_end
            ):
                raise self._format_exception(response)
        elif (
            self.status_code_range_start
            and self.status_code_range_start <= response.status_code
        ):
            raise self._format_exception(response)
        elif (
            self.status_code_range_end
            and self.status_code_range_end >= response.status_code
        ):
            raise self._format_exception(response)

        return response

    @final
    def intercept_response(
        self, resp: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        (self._handle_client_error(resp.r),)
        return resp
