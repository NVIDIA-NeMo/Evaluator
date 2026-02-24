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

"""Centralized default constants for the BYOB subsystem.

All BYOB modules that reference these values should import from here
to avoid duplicated magic numbers across runner.py and compiler.py.
"""

DEFAULT_MAX_TOKENS: int = 4096
DEFAULT_TEMPERATURE: float = 0.0
DEFAULT_TIMEOUT_SECONDS: float = 120.0
