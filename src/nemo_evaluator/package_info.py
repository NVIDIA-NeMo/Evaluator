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

The canonical version is in pyproject.toml; this file mirrors it as
``MAJOR/MINOR/PATCH/PRE_RELEASE`` constants because the FW-CI-templates
``_build_test_publish_wheel.yml`` reusable workflow patches ``PATCH``
in-place for dry-run wheel builds. Removing it would break that build.

When pyproject.toml's ``version`` is bumped, update the constants below
to keep them in sync.
"""

# Below is the _next_ version that will be published, not the currently published one.
MAJOR = 0
MINOR = 4
PATCH = 0
PRE_RELEASE = 'rc0'

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

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
