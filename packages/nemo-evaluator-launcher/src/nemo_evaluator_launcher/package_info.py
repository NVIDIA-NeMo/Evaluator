# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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


# Below is the _next_ version that will be published, not the currently published one.
MAJOR = 0
MINOR = 1
PATCH = 34
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

# BEGIN(if-changed): check the pyproject.toml, too
__package_name__ = "nemo_evaluator_launcher"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://github.com/NVIDIA-NeMo/Eval"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Eval"
__download_url__ = "https://github.com/NVIDIA-NeMo/Evaluator/releases"
__description__ = "Launcher for the evaluations provided by NeMo Evaluator containers with different runtime backends"
__license__ = "Apache2"
__keywords__ = "deep learning, evaluations, machine learning, gpu, NLP, pytorch, torch"
# END(if-changed)
