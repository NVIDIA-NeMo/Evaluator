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
from typing import Any, Dict, List, Optional, cast, final

from flask import Request
from pydantic import BaseModel, Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
)


@register_for_adapter(
    name="payload_modifier",
    description="Modifies request payload by removing, adding, and renaming parameters",
)
@final
class PayloadParamsModifierInterceptor(RequestInterceptor):
    """Adapter for modifying request payload by removing, adding, and renaming parameters"""

    class Params(BaseModel):
        """Configuration parameters for payload modifier interceptor."""

        params_to_remove: Optional[List[str]] = Field(
            default=None, description="List of parameters to remove from payload"
        )
        params_to_add: Optional[Dict[str, Any]] = Field(
            default=None, description="Dictionary of parameters to add to payload"
        )
        params_to_rename: Optional[Dict[str, str]] = Field(
            default=None,
            description="Dictionary mapping old parameter names to new names",
        )

    _params_to_remove: List[str]
    _params_to_add: Dict[str, Any]
    _params_to_rename: Dict[str, str]

    def __init__(self, params: Params):
        """
        Initialize the payload modifier interceptor.

        Args:
            params: Configuration parameters
        """
        self._params_to_remove = params.params_to_remove or []
        self._params_to_add = params.params_to_add or {}
        self._params_to_rename = params.params_to_rename or {}

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest | AdapterResponse:
        # Parse the original request data
        original_data = json.loads(ar.r.get_data())

        # Create a new payload starting with the original
        new_data = original_data.copy()

        # Remove specified parameters
        for param in self._params_to_remove:
            if param in new_data:
                del new_data[param]

        # Add new parameters
        new_data.update(self._params_to_add)

        # Rename parameters
        for old_key, new_key in self._params_to_rename.items():
            if old_key in new_data:
                new_data[new_key] = new_data.pop(old_key)

        # Create new request with modified data
        new_request = cast(
            Request,
            Request.from_values(
                method=ar.r.method,
                headers=dict(ar.r.headers),
                data=json.dumps(new_data),
            ),
        )

        return AdapterRequest(
            r=new_request,
            rctx=ar.rctx,
        )
