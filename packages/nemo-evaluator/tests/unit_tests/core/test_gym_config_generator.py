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

import yaml

from nemo_evaluator.core.gym_config_generator import (
    generate_gym_config,
    write_gym_config,
)


class TestGenerateGymConfig:
    def test_basic_config(self):
        config = generate_gym_config(
            resource_server_host="localhost",
            resource_server_port=9000,
            data_dir="/tmp/test_data",
            eval_type="AIME_2025",
            model_server_url="http://model:8080/v1",
            model_id="my-model",
        )
        assert "nemo_evaluator" in config
        assert "nemo_evaluator_simple_agent" in config
        assert "policy_model" in config

        rs = config["nemo_evaluator"]["resources_servers"]["nemo_evaluator"]
        assert rs["host"] == "localhost"
        assert rs["port"] == 9000

        agent = config["nemo_evaluator_simple_agent"]["responses_api_agents"]["simple_agent"]
        assert agent["resources_server"]["name"] == "nemo_evaluator"
        assert agent["datasets"][0]["jsonl_fpath"] == "/tmp/test_data/AIME_2025.jsonl"

        model = config["policy_model"]["responses_api_models"]["policy_model"]
        assert model["url"] == "http://model:8080/v1"
        assert model["model"] == "my-model"


class TestWriteGymConfig:
    def test_writes_yaml(self, tmp_path):
        config = {"test": "value"}
        path = str(tmp_path / "config.yaml")
        result = write_gym_config(config, output_path=path)
        assert result == path
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"test": "value"}

    def test_auto_generates_path(self):
        config = {"test": "value"}
        path = write_gym_config(config)
        assert path.endswith(".yaml")
        with open(path) as f:
            loaded = yaml.safe_load(f)
        assert loaded == {"test": "value"}
