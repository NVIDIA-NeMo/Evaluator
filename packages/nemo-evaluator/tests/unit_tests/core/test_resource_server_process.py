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

from nemo_evaluator.core.resource_server_process import (
    ResourceServerProcess,
    _find_free_port,
)


class TestFindFreePort:
    def test_returns_valid_port(self):
        port = _find_free_port()
        assert 1024 <= port <= 65535


class TestResourceServerProcess:
    MINIMAL_FRAMEWORK_CONFIG = {
        "framework": {"pkg_name": "simple_evals"},
        "defaults": {},
    }

    def test_init_extracts_harness_package(self):
        rsp = ResourceServerProcess(
            eval_type="AIME_2025",
            framework_config=self.MINIMAL_FRAMEWORK_CONFIG,
        )
        assert rsp.harness_package == "simple_evals"
        assert rsp.eval_type == "AIME_2025"
        assert rsp.host == "localhost"
        assert 1024 <= rsp.port <= 65535

    def test_init_raises_without_pkg_name(self):
        with pytest.raises(ValueError, match="pkg_name"):
            ResourceServerProcess(
                eval_type="AIME_2025",
                framework_config={"framework": {}},
            )

    def test_init_extracts_judge_config(self):
        config = {
            "framework": {"pkg_name": "simple_evals"},
            "defaults": {
                "config": {
                    "params": {
                        "extra": {
                            "judge": {
                                "url": "http://judge:8080/v1",
                                "model_id": "gpt-4",
                                "backend": "openai",
                            }
                        }
                    }
                }
            },
        }
        rsp = ResourceServerProcess(eval_type="AIME_2025", framework_config=config)
        assert rsp.harness_kwargs["judge_config"]["backend"] == "openai"
        assert rsp.harness_kwargs["judge_config"]["model"] == "gpt-4"
        assert rsp.harness_kwargs["judge_config"]["url"] == "http://judge:8080/v1"

    def test_url_property(self):
        rsp = ResourceServerProcess(
            eval_type="AIME_2025",
            framework_config=self.MINIMAL_FRAMEWORK_CONFIG,
            host="0.0.0.0",
            port=12345,
        )
        assert rsp.url == "http://0.0.0.0:12345"
