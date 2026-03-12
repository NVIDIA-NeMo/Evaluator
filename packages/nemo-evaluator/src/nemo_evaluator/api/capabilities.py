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
"""Module for defining benchmark-required capabilities and sample payloads for testing model compatibility."""

from typing import Any, Dict

from pydantic import BaseModel, Field


class Capability(BaseModel):
    name: str = Field(description="Name of the benchmark capability, e.g. 'logprobs'")
    description: str = Field(
        description="Description of the benchmark capability, e.g. 'Uses log probabilities to compute the score'"
    )
    payload: Dict[str, Any] = Field(
        description="Example payload for the benchmark to test model compatibility, e.g. "
        "{'prompt': '3 + 3 = 6', 'max_tokens': 1, 'logprobs': 1, 'echo': True}",
    )


LOGPROBS = Capability(
    name="logprobs",
    description="Uses log probabilities to compute the score",
    payload={
        "prompt": "3 + 3 = 6",
        "max_tokens": 1,
        "logprobs": 1,
        "echo": True,
    },
)

TOOLS = Capability(
    name="tools",
    description="Sends 'tools' input",
    payload={
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": "What is the slope of the line which is perpendicular to the line with the equation y = 3x + 2?",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "find_critical_points",
                    "description": "Finds the critical points of the function. Note that the provided function is in Python 3 syntax.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "function": {
                                "type": "string",
                                "description": "The function to find the critical points for.",
                            },
                            "variable": {
                                "type": "string",
                                "description": "The variable in the function.",
                            },
                            "range": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "The range to consider for finding critical points. Optional. Default is [0.0, 3.4].",
                            },
                        },
                        "required": ["function", "variable"],
                    },
                },
            }
        ],
    },
)

# red circle with black background
_BASE64_IMAGE = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAIAAAACACAIAAABMXPacAAABrUlEQVR4nO3dy07DQBAEQMP//zM5rESCIhQe8XQvVJ2R6J322DmAcxwAAAAAAAAAAAAAAMDuXtIBvu3t0Q/sdaQ90j4c+mf6j1ed8Mdzv1d7ztJgTxz9rcLT1kU6afS3qs78mg7wwcD0x37LF7VcDZGhNBy+YgNSl2TDKuQLyE4h3kG4gPj5j3SGZAEN01+CSWIF9Ex/SeXJFNA2/SWSKlBA5/SX+Wz5T0H/3HQBzZf/MpxwtID+6S+TOd2CwuYK2OXyX8bS2oCwoQL2uvyXmcw2IEwBYRMF7Hj/WQaS24AwBYQpIOz0AvZ9ACxn57cBYQoIU0CYAsIUEKaAMAWEKSBMAWEKCDu9gIa/wf+Ns/PbgDAFhCkgbKKAfR8DA8ltQJgCwoYK2PEuNJPZBoTNFbDXEoyltQFhowXssgSTOac3oL+D4YRuQWGBApqXYD5bZgM6O4ikit2C2jpI5Uk+A3o6CCYJP4QbOshmyH8Kyp4/fgXkCzhyU4hP/+jIcDX2zwQ9x67YgHczc+mZ/lEW5sqbcyt4d3QLb08v8se+PwAAAAAAAAAAAAAAAJ7gArgeJ3hVFTDzAAAAAElFTkSuQmCC"


IMAGE_URL = Capability(
    name="image_url",
    description="Sends image data the the endpoint encoded with base64",
    payload={
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _BASE64_IMAGE,
                        },
                    },
                    {"type": "text", "text": "Describe the image:"},
                ],
            }
        ],
    },
)

IMAGE_URL_WITH_DETAIL = Capability(
    name="image_url_with_detail",
    description='Sends image data the the endpoint encoded with base64 and "detail" parameter (low/high)',
    payload={
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": _BASE64_IMAGE,
                            "detail": "low",
                        },
                    },
                    {"type": "text", "text": "Describe the image:"},
                ],
            }
        ],
    },
)


MULTI_IMAGE_URL = Capability(
    name="multi_image_url",
    description="Sends multiple image data the the endpoint encoded with base64",
    payload={
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Are these images of the same object?"},
                    {"type": "image_url", "image_url": {"url": _BASE64_IMAGE}},
                    {"type": "image_url", "image_url": {"url": _BASE64_IMAGE}},
                ],
            },
        ],
        "max_tokens": 256,
    },
)

# speaker is saying "dzień dobry" ("good morning" in polish)
_BASE64_AUDIO = "data:audio/mp3;base64,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"
AUDIO_URL = Capability(
    name="audio_url",
    description="Sends audio data the the endpoint encoded with base64",
    payload={
        "max_tokens": 256,
        "messages": [
            {
                "content": [
                    {
                        "audio_url": {
                            "url": _BASE64_AUDIO,
                        },
                        "type": "audio_url",
                    },
                    {
                        "text": "Please recognize the speech and only output the recognized content:",
                        "type": "text",
                    },
                ],
                "role": "user",
            }
        ],
    },
)

# full screen is red and then it switches to green
_BASE64_VIDEO = "data:video/mp4;base64,AAAAHGZ0eXBpc29tAAACAGlzb21pc28ybXA0MQAAAAhmcmVlAAAPkG1kYXQAAAGzABAHAAABthGBxhQbYLIHfG238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt+AAABtlPAz//////////////////////////////////////////7AAABtlWAz//////////////////////////////////////////7AAABtlfAz//////////////////////////////////////////7AAABtlmAz//////////////////////////////////////////7AAABtmjgZ//////////////////////////////////////////9AAABtlOAz//////////////////////////////////////////7AAABtlXAz//////////////////////////////////////////7AAABtleAz//////////////////////////////////////////7AAABtlnAz//////////////////////////////////////////7AAABtmjAZ//////////////////////////////////////////9AAABtlPAz//////////////////////////////////////////7AAABswAQhwAAAbYVgcYUG2CyB3xtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfgAAAbZXwM//////////////////////////////////////////+wAAAbZZgM//////////////////////////////////////////+wAAAbZo4Gf//////////////////////////////////////////QAAAbMAEMcAAAG2E4HGMG2BaAUMbbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G234AAAG2VcDP//////////////////////////////////////////sAAAG2V4DP//////////////////////////////////////////sAAAG2WcDP//////////////////////////////////////////sAAAG2aMBn//////////////////////////////////////////0AAAG2U8DP//////////////////////////////////////////sAAAG2VYDP//////////////////////////////////////////sAAAG2V8DP//////////////////////////////////////////sAAAG2WYDP//////////////////////////////////////////sAAAG2aOBn//////////////////////////////////////////0AAAG2U4DP//////////////////////////////////////////sAAAG2VcDP//////////////////////////////////////////sAAAGzABFHAAABtheBxjBtgWgFDG238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt/G238bbfxtt+AAABtlnAz//////////////////////////////////////////7AAABtmjAZ//////////////////////////////////////////9AAABtlPAz//////////////////////////////////////////7AAAD0W1vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAABkAAAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAL7dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAABkAAAAAAAAAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAEAAAABAAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAZAAAAAAAAAQAAAAACc21kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAKAAAAQAAVcQAAAAAAC1oZGxyAAAAAAAAAAB2aWRlAAAAAAAAAAAAAAAAVmlkZW9IYW5kbGVyAAAAAh5taW5mAAAAFHZtaGQAAAABAAAAAAAAAAAAAAAkZGluZgAAABxkcmVmAAAAAAAAAAEAAAAMdXJsIAAAAAEAAAHec3RibAAAANpzdHNkAAAAAAAAAAEAAADKbXA0dgAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAAEAAQAASAAAAEgAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAABj//wAAAGBlc2RzAAAAAAOAgIBPAAEABICAgEEgEQAAAAAKAAAAABNqBYCAgC8AAAGwAQAAAbWJEwAAAQAAAAEgAMSNiAAtCAQgFGMAAAGyTGF2YzU5LjM3LjEwMAaAgIABAgAAABRidHJ0AAAAAAAKAAAAABNqAAAAGHN0dHMAAAAAAAAAAQAAACAAAAgAAAAAIHN0c3MAAAAAAAAABAAAAAEAAAANAAAAEQAAAB0AAAAcc3RzYwAAAAAAAAABAAAAAQAAACAAAAABAAAAlHN0c3oAAAAAAAAAAAAAACAAAALRAAAAJwAAACcAAAAnAAAAJwAAACcAAAAnAAAAJwAAACcAAAAnAAAAJwAAACcAAALRAAAAJwAAACcAAAAnAAAC0QAAACcAAAAnAAAAJwAAACcAAAAnAAAAJwAAACcAAAAnAAAAJwAAACcAAAAnAAAC0QAAACcAAAAnAAAAJwAAABRzdGNvAAAAAAAAAAEAAAAsAAAAYnVkdGEAAABabWV0YQAAAAAAAAAhaGRscgAAAAAAAAAAbWRpcmFwcGwAAAAAAAAAAAAAAAAtaWxzdAAAACWpdG9vAAAAHWRhdGEAAAABAAAAAExhdmY1OS4yNy4xMDA="

VIDEO_URL = Capability(
    name="video_url",
    description="Sends video data the the endpoint encoded with base64",
    payload={
        "max_tokens": 256,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe the video"},
                    {
                        "type": "video_url",
                        "video_url": {"url": _BASE64_VIDEO},
                    },
                ],
            }
        ],
    },
)

BENCHMARK_CAPABILITIES: Dict[str, Capability] = {
    capability.name: capability
    for capability in [
        LOGPROBS,
        TOOLS,
        IMAGE_URL,
        IMAGE_URL_WITH_DETAIL,
        MULTI_IMAGE_URL,
        VIDEO_URL,
        AUDIO_URL,
    ]
}
