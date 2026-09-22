# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from urllib.parse import urlparse

NVCF_HOSTNAMES = frozenset({"integrate.api.nvidia.com", "api.nvcf.nvidia.com"})
DEFAULT_NVCF_POLL_SECONDS = "1800"


def is_nvcf_url(url: str | None) -> bool:
    if not url:
        return False
    hostname = urlparse(url).hostname or ""
    return hostname in NVCF_HOSTNAMES
