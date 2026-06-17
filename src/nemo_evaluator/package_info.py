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
"""Package version constants.

Version is derived from git tags via hatch-vcs (PEP 440 local version,
e.g. ``0.3.0.post5+g1a2b3c4`` for untagged commits). MAJOR/MINOR/PATCH
are kept for FW-CI-templates compatibility.
"""

from importlib.metadata import version as _v, PackageNotFoundError as _E

MAJOR = 0
MINOR = 3
PATCH = 1
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

try:
    __version__ = _v("nemo-evaluator")
except _E:
    __version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__shortversion__ = __version__.split("+")[0]

__package_name__ = "nemo_evaluator"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/Evaluator"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Evaluator"
__download_url__ = "https://github.com/NVIDIA-NeMo/Evaluator/releases"
__description__ = (
    "NeMo Evaluator — benchmark environments, pluggable solvers, interceptor proxy, and decision-grade scoring for LLMs"
)
__license__ = "Apache-2.0"
__keywords__ = "deep learning, machine learning, NLP, evaluation, benchmarks, llm"
