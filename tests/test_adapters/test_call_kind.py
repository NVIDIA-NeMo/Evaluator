# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from nemo_evaluator.adapters.call_kind import (
    NEL_CALL_KIND_COMPACTION,
    NEL_CALL_KIND_HEADER,
    is_compaction_call_kind,
    merge_compaction_extra_headers,
    strip_call_kind_header,
)


def test_merge_compaction_extra_headers_sets_marker():
    out = merge_compaction_extra_headers({"temperature": 0.2})
    assert out["temperature"] == 0.2
    assert out["extra_headers"][NEL_CALL_KIND_HEADER] == NEL_CALL_KIND_COMPACTION


def test_merge_compaction_extra_headers_preserves_existing():
    out = merge_compaction_extra_headers({"extra_headers": {"X-Keep": "1"}})
    assert out["extra_headers"]["X-Keep"] == "1"
    assert out["extra_headers"][NEL_CALL_KIND_HEADER] == NEL_CALL_KIND_COMPACTION


def test_strip_call_kind_header_is_case_insensitive():
    headers = {"X-NEL-Call-Kind": "compaction", "content-type": "application/json"}
    kind = strip_call_kind_header(headers)
    assert is_compaction_call_kind(kind)
    assert "X-NEL-Call-Kind" not in headers
    assert headers["content-type"] == "application/json"
