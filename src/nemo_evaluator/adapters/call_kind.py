# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any, Mapping, MutableMapping

NEL_CALL_KIND_HEADER = "x-nel-call-kind"
NEL_CALL_KIND_COMPACTION = "compaction"


def is_compaction_call_kind(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() == NEL_CALL_KIND_COMPACTION


def strip_call_kind_header(headers: MutableMapping[str, str]) -> str | None:
    kind = None
    for key in list(headers):
        if key.lower() == NEL_CALL_KIND_HEADER:
            kind = headers.pop(key)
    return kind


def merge_compaction_extra_headers(kwargs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    out = dict(kwargs or {})
    existing = out.get("extra_headers")
    headers = dict(existing) if isinstance(existing, Mapping) else {}
    headers[NEL_CALL_KIND_HEADER] = NEL_CALL_KIND_COMPACTION
    out["extra_headers"] = headers
    return out
