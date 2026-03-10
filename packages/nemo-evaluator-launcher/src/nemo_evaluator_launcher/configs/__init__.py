# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from omegaconf import OmegaConf


def select_not_null(key: str, default: str, *, _root_: object) -> str:
    """Like oc.select but also falls back to default when the value is null."""
    value = OmegaConf.select(
        _root_, key, default=None, throw_on_resolution_failure=False
    )
    return default if value is None else value


OmegaConf.register_new_resolver("select_not_null", select_not_null, use_cache=False)
