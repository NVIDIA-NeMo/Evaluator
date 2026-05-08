# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import logging
from typing import Any

from nemo_evaluator.adapters.types import AdapterRequest, RequestInterceptor

logger = logging.getLogger(__name__)

# Transport-level headers that must not be set by user config — they are
# managed by the HTTP layer and overriding them breaks framing or routing.
# Authorization is intentionally NOT included so users can override auth for
# inference-gateway style deployments.
_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset({"host", "content-length", "connection", "transfer-encoding"})


def _remove_keys(obj: Any, keys: set[str]) -> Any:
    if isinstance(obj, dict):
        return {k: _remove_keys(v, keys) for k, v in obj.items() if k not in keys}
    if isinstance(obj, list):
        return [_remove_keys(item, keys) for item in obj]
    return obj


def _rename_keys(obj: Any, mapping: dict[str, str]) -> Any:
    if isinstance(obj, dict):
        return {mapping.get(k, k): _rename_keys(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_keys(item, mapping) for item in obj]
    return obj


class Interceptor(RequestInterceptor):
    def __init__(
        self,
        *,
        params_to_remove: list[str] | None = None,
        params_to_add: dict[str, Any] | None = None,
        params_to_rename: dict[str, str] | None = None,
        headers_to_add: dict[str, str] | None = None,
        headers_to_remove: list[str] | None = None,
        headers_to_rename: dict[str, str] | None = None,
    ) -> None:
        self._remove = set(params_to_remove or [])
        self._add: dict[str, Any] = params_to_add or {}
        self._rename: dict[str, str] = params_to_rename or {}

        self._headers_add: dict[str, str] = {}
        for name, value in (headers_to_add or {}).items():
            if name.lower() in _HOP_BY_HOP_HEADERS:
                logger.warning("Dropping hop-by-hop header from headers_to_add: %s", name)
                continue
            self._headers_add[name] = value

        self._headers_remove_lc: set[str] = {h.lower() for h in (headers_to_remove or [])}

        # Validate rename targets the same way headers_to_add validates names —
        # otherwise users could rename an arbitrary header *into* a hop-by-hop
        # name and bypass the guard above.
        self._headers_rename_lc: dict[str, str] = {}
        for old, new in (headers_to_rename or {}).items():
            if new.lower() in _HOP_BY_HOP_HEADERS:
                logger.warning("Dropping hop-by-hop header from headers_to_rename target: %s -> %s", old, new)
                continue
            self._headers_rename_lc[old.lower()] = new

        self._logged_once = False

    async def intercept_request(self, req: AdapterRequest) -> AdapterRequest:
        if self._remove:
            req.body = _remove_keys(req.body, self._remove)

        if self._rename:
            req.body = _rename_keys(req.body, self._rename)

        if self._add:
            req.body.update(self._add)

        if self._headers_remove_lc:
            req.headers = {k: v for k, v in req.headers.items() if k.lower() not in self._headers_remove_lc}

        if self._headers_rename_lc:
            renamed: dict[str, str] = {}
            for k, v in req.headers.items():
                new_name = self._headers_rename_lc.get(k.lower())
                renamed[new_name if new_name is not None else k] = v
            req.headers = renamed

        if self._headers_add:
            add_lc = {k.lower() for k in self._headers_add}
            req.headers = {k: v for k, v in req.headers.items() if k.lower() not in add_lc}
            req.headers.update(self._headers_add)

        if not self._logged_once:
            logger.info(
                "payload_modifier: remove=%s, add=%s, rename=%s, headers_add=%s, headers_remove=%s, headers_rename=%s",
                sorted(self._remove),
                sorted(self._add),
                sorted(self._rename),
                sorted(self._headers_add),
                sorted(self._headers_remove_lc),
                sorted(self._headers_rename_lc),
            )
            self._logged_once = True

        return req
