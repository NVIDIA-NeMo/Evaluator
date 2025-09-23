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

import pytest

from nemo_evaluator.adapters.interceptors.raise_client_error_interceptor import (
    RaiseClientErrorInterceptor,
)
from nemo_evaluator.adapters.types import AdapterResponse


@pytest.mark.parametrize(
    "status_code,should_raise",
    [(400, True), (404, True), (499, True), (200, False), (300, False), (500, False)],
)
def test_raise_client_errors(
    status_code,
    should_raise,
    adapter_global_context,
    adapter_request_context,
    create_response,
):
    params = RaiseClientErrorInterceptor.Params()
    interceptor = RaiseClientErrorInterceptor(params)
    mock_response = AdapterResponse(
        r=create_response(status_code), rctx=adapter_request_context
    )

    if should_raise:
        with pytest.raises(Exception, match="Upstream endpoint error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)
    else:
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response


@pytest.mark.parametrize(
    "status_code,should_raise", [(400, False), (429, False), (404, True), (499, True)]
)
def test_exclude_status_codes(
    status_code,
    should_raise,
    adapter_global_context,
    adapter_request_context,
    create_response,
):
    params = RaiseClientErrorInterceptor.Params(exclude_status_codes=[400, 429])
    interceptor = RaiseClientErrorInterceptor(params)
    mock_response = AdapterResponse(
        r=create_response(status_code), rctx=adapter_request_context
    )

    if should_raise:
        with pytest.raises(Exception, match="Upstream endpoint error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)
    else:
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response


@pytest.mark.parametrize(
    "status_code,should_raise", [(333, True), (400, False), (50, True), (150, False)]
)
def test_custom_ranges(
    status_code,
    should_raise,
    adapter_global_context,
    adapter_request_context,
    create_response,
):
    params = RaiseClientErrorInterceptor.Params(
        status_codes=[333], status_code_range_start=0, status_code_range_end=100
    )
    interceptor = RaiseClientErrorInterceptor(params)
    mock_response = AdapterResponse(
        r=create_response(status_code), rctx=adapter_request_context
    )

    if should_raise:
        with pytest.raises(Exception, match="Upstream endpoint error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)
    else:
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response
