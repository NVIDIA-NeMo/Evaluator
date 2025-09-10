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
import requests

from nemo_evaluator.adapters.interceptors import RaiseClientErrorInterceptor
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequestContext,
    AdapterResponse,
)


@pytest.fixture
def adapter_global_context():
    return AdapterGlobalContext(output_dir="/tmp", url="http://localhost")


@pytest.fixture
def adapter_request_context():
    return AdapterRequestContext()


@pytest.fixture
def create_response():
    def _create_response(status_code: int):
        response = requests.Response()
        response.status_code = status_code
        return response

    return _create_response


@pytest.mark.parametrize(
    "desc,params,exception_substr",
    [
        (
            "overlap",
            RaiseClientErrorInterceptor.Params(
                exclude_status_codes=[499, 500],
                status_codes=[499],
            ),
            "status code.+ cannot be listed in both",
        ),
        (
            "invalid range",
            RaiseClientErrorInterceptor.Params(
                status_code_range_start=500,
                status_code_range_end=400,
            ),
            "Status code start and end is not a valid range",
        ),
    ],
)
def test_raise_client_errors_validation(
    desc: str, params: RaiseClientErrorInterceptor.Params, exception_substr: str
):
    with pytest.raises(Exception, match=exception_substr):
        RaiseClientErrorInterceptor(params)


@pytest.mark.parametrize(
    "status_code",
    [
        (200),
        (204),
        (408),
        (429),
        (500),
        (503),
    ],
)
def test_raise_client_errors_pass(
    adapter_global_context, adapter_request_context, create_response, status_code
):
    # Given: A response with non-4xx status code, except for default exclude_status_codes
    interceptor = RaiseClientErrorInterceptor(RaiseClientErrorInterceptor.Params())
    mock_response = AdapterResponse(
        r=create_response(status_code),
        rctx=adapter_request_context,
    )

    # When: The response is intercepted
    result = interceptor.intercept_response(mock_response, adapter_global_context)

    # Then: The response should not be modified
    assert result == mock_response


@pytest.mark.parametrize(
    "status_code",
    [
        (400),
        (404),
        (499),
    ],
)
def test_raise_client_errors(
    adapter_global_context, adapter_request_context, create_response, status_code
):
    # Given: A response with 4xx status code
    interceptor = RaiseClientErrorInterceptor(RaiseClientErrorInterceptor.Params())
    mock_response = AdapterResponse(
        r=create_response(status_code),
        rctx=adapter_request_context,
    )

    # When: The response is intercepted with 4xx status code
    # Then: An exception is raised
    with pytest.raises(Exception, match="Client error detected"):
        interceptor.intercept_response(mock_response, adapter_global_context)


def test_raise_client_errors_params(
    subtests, adapter_global_context, adapter_request_context, create_response
):
    # Given: A response with 4xx status code
    params = RaiseClientErrorInterceptor.Params(
        exclude_status_codes=None,
        status_codes=[333],
        status_code_range_start=0,
        status_code_range_end=100,
    )
    interceptor = RaiseClientErrorInterceptor(params)

    with subtests.test(msg="out of range"):
        mock_response = AdapterResponse(
            r=create_response(400),
            rctx=adapter_request_context,
        )
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response

    with subtests.test(msg="empty exclude"):
        mock_response = AdapterResponse(
            r=create_response(429),
            rctx=adapter_request_context,
        )
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response

    with subtests.test(msg="status_codes raises"):
        mock_response = AdapterResponse(
            r=create_response(333),
            rctx=adapter_request_context,
        )
        with pytest.raises(Exception, match="Client error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)

    with subtests.test(msg="custom status_code_range raises"):
        mock_response = AdapterResponse(
            r=create_response(50),
            rctx=adapter_request_context,
        )
        with pytest.raises(Exception, match="Client error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)


def test_raise_client_errors_start_range_only(
    subtests, adapter_global_context, adapter_request_context, create_response
):
    # Given: A response with 4xx status code
    params = RaiseClientErrorInterceptor.Params(
        status_code_range_start=500,
        status_code_range_end=None,
    )
    interceptor = RaiseClientErrorInterceptor(params)

    with subtests.test(msg="out of start range"):
        mock_response = AdapterResponse(
            r=create_response(499),
            rctx=adapter_request_context,
        )
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response

    with subtests.test(msg="within start range"):
        mock_response = AdapterResponse(
            r=create_response(1000),
            rctx=adapter_request_context,
        )
        with pytest.raises(Exception, match="Client error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)


def test_raise_client_errors_end_range_only(
    subtests, adapter_global_context, adapter_request_context, create_response
):
    # Given: A response with 4xx status code
    params = RaiseClientErrorInterceptor.Params(
        status_code_range_start=None,
        status_code_range_end=500,
    )
    interceptor = RaiseClientErrorInterceptor(params)

    with subtests.test(msg="out of end range"):
        mock_response = AdapterResponse(
            r=create_response(1000),
            rctx=adapter_request_context,
        )
        result = interceptor.intercept_response(mock_response, adapter_global_context)
        assert result == mock_response

    with subtests.test(msg="within start range"):
        mock_response = AdapterResponse(
            r=create_response(1),
            rctx=adapter_request_context,
        )
        with pytest.raises(Exception, match="Client error detected"):
            interceptor.intercept_response(mock_response, adapter_global_context)
