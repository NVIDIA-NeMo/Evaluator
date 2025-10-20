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
"""Printing utils for more structured or visually appealing prints.

NOTE: use printing only for main application output that matters. For logging,
see `logging_utils.py`.

USAGE:
```
  from nemo_evaluator_launcher.common.printing_utils import red, bold
  print(bold(red("some red bold")))
```


"""

import os

# If this env var is set, it will override a more standard "LOG_LEVEL". If
# both are unset, default would be used.
_DISABLE_COLOR_ENV_VAR = "NEMO_EVALUATOR_DISABLE_COLOR"


def _is_color_disabled():
    env_var = os.environ.get(_DISABLE_COLOR_ENV_VAR, "0").lower()

    if "1" in env_var or "yes" in env_var or "y" in env_var or "true" in env_var:
        return True

    return False


_CODES: dict[str, str] = dict(
    green="\033[32m",
    red="\033[31m",
    red_bg="\033[41m",  # red background
    cyan="\033[36m",
    yellow="\033[33m",
    magenta="\033[35m",
    grey="\033[90m",
    bold="\033[1m",
    reset="\033[0m",
)

# If the colors are disabled, we null-out all the codes.
if _is_color_disabled():
    for c in _CODES.keys():
        _CODES[c] = ""


def green(s: str) -> str:
    return _CODES["green"] + s + _CODES["reset"]


def red(s: str) -> str:
    return _CODES["red"] + s + _CODES["reset"]


def red_bg(s: str) -> str:
    return _CODES["red_bg"] + s + _CODES["reset"]


def cyan(s: str) -> str:
    return _CODES["cyan"] + s + _CODES["reset"]


def yellow(s: str) -> str:
    return _CODES["yellow"] + s + _CODES["reset"]


def magenta(s: str) -> str:
    return _CODES["magenta"] + s + _CODES["reset"]


def grey(s: str) -> str:
    return _CODES["grey"] + s + _CODES["reset"]


def bold(s: str) -> str:
    return _CODES["bold"] + s + _CODES["reset"]
