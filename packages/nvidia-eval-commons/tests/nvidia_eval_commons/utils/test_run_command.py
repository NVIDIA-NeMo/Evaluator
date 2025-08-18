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

import pytest
from nvidia_eval_commons.core.utils import run_command


def test_run_command_propagates_error_output():
    """When propagate_errors=True and the command fails, a RuntimeError containing
    the subprocess output should be raised.
    """

    failing_script = 'echo "STDOUT ok"; echo "STDERR boom" 1>&2; exit 7'

    with pytest.raises(RuntimeError) as excinfo:
        run_command(failing_script, propagate_errors=True)

    message = str(excinfo.value)
    assert "Evaluation failed" in message
    # Output is captured via a PTY; both stdout and stderr are merged. Ensure our marker is present.
    assert "STDERR boom" in message
