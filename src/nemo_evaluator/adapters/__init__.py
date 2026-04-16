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
"""Adapter proxy — interceptor-based HTTP middleware replacing LiteLLM.

Public API:
    start_adapter_proxy(...)  — start a local proxy server
    AdapterPipeline           — the async interceptor chain
    InterceptorRegistry       — name→class resolution
"""

from nemo_evaluator.adapters.pipeline import AdapterPipeline
from nemo_evaluator.adapters.proxy import ProxyHandle, start_adapter_proxy
from nemo_evaluator.adapters.registry import InterceptorRegistry
from nemo_evaluator.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    InterceptorContext,
    RequestInterceptor,
    RequestToResponseInterceptor,
    ResponseInterceptor,
)

__all__ = [
    "AdapterPipeline",
    "AdapterRequest",
    "AdapterResponse",
    "InterceptorContext",
    "InterceptorRegistry",
    "ProxyHandle",
    "RequestInterceptor",
    "RequestToResponseInterceptor",
    "ResponseInterceptor",
    "start_adapter_proxy",
]
