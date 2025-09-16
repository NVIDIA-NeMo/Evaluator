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
"""End-to-end tests for eval-factory using fake endpoint."""

import pytest
from nemo_evaluator.logging.utils import logger


class TestEvalFactoryFakeEndpoint:
    """Test eval-factory integration with fake endpoint."""

    def test_fake_endpoint_chat_completion(self, fake_endpoint):
        """Test that the fake endpoint responds to chat completion requests."""
        import requests

        # Test request based on the logs
        test_request = {
            "model": "Qwen/Qwen3-8B",
            "messages": [
                {
                    "role": "user",
                    "content": "\nAnswer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D/E/F/G/H/I/J' (e.g. 'Answer: A'). \n\nTypical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.\n\nA) Safe practices, Fear, Jealousy, Trivial\nB) Unsafe practices, Distress, Joy, Trivial\nC) Safe practices, Wants, Jealousy, Trivial\nD) Safe practices, Distress, Fear, Trivial\nE) Unsafe practices, Wants, Jealousy, Serious\nF) Safe practices, Distress, Jealousy, Serious\nG) Safe practices, Wants, Fear, Serious\nH) Unsafe practices, Wants, Fear, Trivial\nI) Unsafe practices, Distress, Fear, Serious\nJ) None\n",
                }
            ],
            "temperature": 0.0,
            "max_tokens": 4096,
            "top_p": 1e-05,
            "seed": 2781248596,
        }

        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert "message" in data["choices"][0]
        assert "content" in data["choices"][0]["message"]

        # Check content contains expected answer
        content = data["choices"][0]["message"]["content"]
        assert "Answer: I" in content

        # Check endpoint headers
        assert "Status" in response.headers
        assert "Reqid" in response.headers

        logger.info("✅ Fake endpoint chat completion test passed")

    @pytest.mark.parametrize(
        "name,content",
        [
            (
                "Advertising Regulatory",
                "Typical advertising regulatory bodies suggest, for example that adverts must not: encourage _________, cause unnecessary ________ or _____, and must not cause _______ offence.",
            ),
            (
                "Downsizing Issues",
                "There are two main issues associated with _____ sizing. _______ is a key issue as due to the information policy of the corporation it can be argued that employees have a right to know if they are being made redundant.",
            ),
            (
                "Managers and Shareholders",
                "Managers are entrusted to run the company in the best interest of ________. Specifically, they have a duty to act for the benefit of the company, as well as a duty of ________ and of ________.",
            ),
        ],
    )
    def test_multiple_question_types(self, fake_endpoint, name, content):
        """Test that the fake endpoint handles different question types correctly."""
        import requests

        test_request = {
            "model": "Qwen/Qwen3-8B",
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "max_tokens": 100,
        }

        response = requests.post(
            "http://localhost:8000/v1/chat/completions",
            json=test_request,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        assert response.status_code == 200
        data = response.json()
        assert "choices" in data
        assert len(data["choices"]) > 0

        logger.info(f"✅ {name} question handled correctly")
