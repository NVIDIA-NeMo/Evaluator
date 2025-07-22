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

import logging
import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent.parent.parent / "scripts" / "snippets"
print(SCRIPT_DIR)

# TODO: we skip the safety-harness and simple-evals because they require access
# to the judge endpoint. We should use a mock server for these tests.
SCRITPS = sorted(
    [
        path
        for path in SCRIPT_DIR.glob("*.py")
        if path.name in ("arc_challenge.py", "bfcl.py", "bigcode.py", "garak.py", "lambada.py")
    ]
)


@pytest.fixture(scope="module")
def deployment_process():
    # Run deployment
    deploy_proc = subprocess.Popen(
        [
            "python",
            "tests/functional_tests/deploy_in_fw_script.py",
            "--nemo2_ckpt_path",
            "/checkpoints/llama-3_2-1b-instruct_v2.0",
            "--max_batch_size",
            "4",
            "--port",
            "8080",
        ]
    )

    yield deploy_proc

    deploy_proc.send_signal(signal.SIGINT)
    try:
        deploy_proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(deploy_proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    finally:
        subprocess.run(["pkill", f"-{signal.SIGTERM}", "tritonserver"], check=False)


@pytest.fixture(autouse=True)
def cleanup_results():
    """Clean up results directory after each test."""
    yield
    results_dir = "/results"
    if os.path.exists(results_dir):
        logger.info(f"Cleaning up results directory: {results_dir}")
        shutil.rmtree(results_dir)


@pytest.mark.parametrize("script_path", SCRITPS)
def test_snippet(deployment_process, script_path):
    eval_proc = subprocess.run(["python", script_path], capture_output=True)
    assert eval_proc.returncode == 0
    stdout = eval_proc.stdout.decode("utf-8")
    for msg in ("Output reading completed.", "Subprocess finished with return code: 0", "tasks", "metrics", "scores"):
        assert msg in stdout
