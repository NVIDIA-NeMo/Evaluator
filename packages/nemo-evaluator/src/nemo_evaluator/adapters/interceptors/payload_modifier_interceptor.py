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

"""Payload modifier interceptor that modifies request payloads."""

import json
from typing import Any, Dict, List, Optional, cast, final

from flask import Request
from pydantic import Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterRequest,
    AdapterResponse,
    RequestInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@register_for_adapter(
    name="payload_modifier",
    description="Modifies request payload by removing, adding, and renaming parameters",
)
@final
class PayloadParamsModifierInterceptor(RequestInterceptor):
    """Adapter for modifying request payload by removing, adding, and renaming parameters"""

    class Params(BaseLoggingParams):
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

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Payload modifier interceptor initialized",
            params_to_remove=self._params_to_remove,
            params_to_add=(
                list(self._params_to_add.keys()) if self._params_to_add else []
            ),
            params_to_rename=(
                list(self._params_to_rename.keys()) if self._params_to_rename else []
            ),
        )

    @final
    def intercept_request(
        self, ar: AdapterRequest, context: AdapterGlobalContext
    ) -> AdapterRequest | AdapterResponse:
        # Parse the original request data
        original_data = json.loads(ar.r.get_data())

        self.logger.debug(
            "Processing request payload",
            original_keys=(
                list(original_data.keys())
                if isinstance(original_data, dict)
                else "unknown"
            ),
            params_to_remove=self._params_to_remove,
            params_to_add=(
                list(self._params_to_add.keys()) if self._params_to_add else []
            ),
            params_to_rename=(
                list(self._params_to_rename.keys()) if self._params_to_rename else []
            ),
        )

        # Create a new payload starting with the original
        new_data = original_data.copy()

        # Remove specified parameters
        def _remove_param_recursively(data: Any, param: str, _parent_key: str = ""):
            if isinstance(data, dict):
                if param in data:
                    del data[param]
                    self.logger.debug(
                        "Removed parameter", parameter=param, parent_key=_parent_key
                    )
                for key, value in data.items():
                    _remove_param_recursively(
                        value,
                        param,
                        _parent_key=f"{_parent_key}.{key}" if _parent_key else key,
                    )
            elif isinstance(data, list):
                for idx, item in enumerate(data):
                    _remove_param_recursively(
                        item,
                        param,
                        _parent_key=f"{_parent_key}[{idx}]",
                    )

        for param in self._params_to_remove:
            _remove_param_recursively(new_data, param)

        # Add new parameters
        new_data.update(self._params_to_add)
        if self._params_to_add:
            self.logger.debug(
                "Added parameters", parameters=list(self._params_to_add.keys())
            )

        # Rename parameters
        for old_key, new_key in self._params_to_rename.items():
            if old_key in new_data:
                new_data[new_key] = new_data.pop(old_key)
                self.logger.debug("Renamed parameter", old_key=old_key, new_key=new_key)

        # Create new request with modified data
        new_request = cast(
            Request,
            Request.from_values(
                method=ar.r.method,
                headers=dict(ar.r.headers),
                data=json.dumps(new_data),
            ),
        )

        self.logger.info(
            "Request payload modified",
            original_keys_count=(
                len(original_data.keys()) if isinstance(original_data, dict) else 0
            ),
            modified_keys_count=(
                len(new_data.keys()) if isinstance(new_data, dict) else 0
            ),
            modifications_made=len(self._params_to_remove)
            + len(self._params_to_add)
            + len(self._params_to_rename),
        )

        return AdapterRequest(
            r=new_request,
            rctx=ar.rctx,
        )
