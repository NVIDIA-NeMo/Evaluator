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

from typing import final

import requests
from pydantic import BaseModel

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestToResponseInterceptor,
)


@register_for_adapter(
    name="endpoint",
    description="Makes the actual request to the upstream API",
)
@final
class EndpointInterceptor(RequestToResponseInterceptor):
    """Makes the actual request to the upstream API."""

    class Params(BaseModel):
        """Configuration parameters for endpoint interceptor."""

        pass

    def __init__(self, params: Params):
        """
        Initialize the endpoint interceptor.

        Args:
            params: Configuration parameters
        """
        return

    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterResponse:
        """Make the actual request to the upstream API.

        Args:
            ar: The adapter request
            context: Global context containing server-level configuration

        Returns:
            AdapterResponse with the response from the upstream API
        """
        # This is a final interceptor, we'll need the flask_request and api
        resp = AdapterResponse(
            r=requests.request(
                method=ar.r.method,
                url=context.url,
                headers={k: v for k, v in ar.r.headers if k.lower() != "host"},
                json=ar.r.json,
                cookies=ar.r.cookies,
                allow_redirects=False,
            ),
            rctx=ar.rctx,
        )
        return resp
