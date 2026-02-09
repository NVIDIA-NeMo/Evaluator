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

"""Test module for image data sanitization in logging."""

import base64

from nemo_evaluator.adapters.interceptors.caching_interceptor import CachingInterceptor


class TestImageLoggingSanitization:
    """Test that image data is properly sanitized in logs."""

    def test_sanitize_base64_image_in_image_url(self):
        """Test that base64 image data in image_url field is sanitized."""
        # Create a small test image (1x1 pixel PNG)
        small_png = base64.b64encode(
            b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\x00\x01\x00"
            b"\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        ).decode("utf-8")

        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{small_png}"},
                        },
                        {"type": "text", "text": "What is in this image?"},
                    ],
                }
            ],
            "model": "test-model",
        }

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        # Verify that the image data is replaced with a brief description
        assert "messages" in sanitized
        assert len(sanitized["messages"]) == 1
        assert "content" in sanitized["messages"][0]
        assert len(sanitized["messages"][0]["content"]) == 2

        image_content = sanitized["messages"][0]["content"][0]
        assert "image_url" in image_content
        assert "url" in image_content["image_url"]

        # The URL should be sanitized to a brief description
        url = image_content["image_url"]["url"]
        assert url.startswith("<image:")
        assert "format=png" in url
        assert "size≈" in url
        assert "bytes>" in url

        # The base64 data should NOT be in the sanitized output
        assert small_png not in str(sanitized)

        # The text content should remain unchanged
        text_content = sanitized["messages"][0]["content"][1]
        assert text_content["type"] == "text"
        assert text_content["text"] == "What is in this image?"

    def test_sanitize_base64_image_different_formats(self):
        """Test sanitization works for different image formats."""
        formats = ["png", "jpeg", "jpg", "gif", "webp"]

        for fmt in formats:
            request_data = {
                "image_url": {"url": f"data:image/{fmt};base64,iVBORw0KGgoAAAANSUhEUg"}
            }

            sanitized = CachingInterceptor.sanitize_request_data_for_logging(
                request_data
            )

            url = sanitized["image_url"]["url"]
            assert url.startswith("<image:")
            assert f"format={fmt}" in url
            assert "size≈" in url

    def test_sanitize_preserves_non_image_data(self):
        """Test that non-image data is preserved during sanitization."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello, world!"},
                    ],
                }
            ],
            "model": "test-model",
            "temperature": 0.7,
            "max_tokens": 100,
        }

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        # All non-image data should be preserved
        assert sanitized == request_data

    def test_sanitize_nested_structures(self):
        """Test sanitization works with deeply nested structures."""
        request_data = {
            "level1": {
                "level2": {
                    "level3": {
                        "image_url": {
                            "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"
                        }
                    }
                }
            }
        }

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        url = sanitized["level1"]["level2"]["level3"]["image_url"]["url"]
        assert url.startswith("<image:")
        assert "format=png" in url

    def test_sanitize_multiple_images(self):
        """Test sanitization works when there are multiple images."""
        request_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"
                            },
                        },
                        {"type": "text", "text": "Compare these images:"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA"
                            },
                        },
                    ],
                }
            ]
        }

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        # Both images should be sanitized
        content = sanitized["messages"][0]["content"]
        assert "<image:" in content[0]["image_url"]["url"]
        assert "format=png" in content[0]["image_url"]["url"]
        assert "<image:" in content[2]["image_url"]["url"]
        assert "format=jpeg" in content[2]["image_url"]["url"]

        # Text should be preserved
        assert content[1]["text"] == "Compare these images:"

    def test_sanitize_handles_list_of_dicts(self):
        """Test that sanitization works with lists of dictionaries."""
        request_data = [
            {"image_url": {"url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg"}},
            {"text": "Some text"},
            {"image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA"}},
        ]

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        assert isinstance(sanitized, list)
        assert len(sanitized) == 3
        assert "<image:" in sanitized[0]["image_url"]["url"]
        assert sanitized[1]["text"] == "Some text"
        assert "<image:" in sanitized[2]["image_url"]["url"]

    def test_sanitize_non_base64_urls_preserved(self):
        """Test that non-base64 URLs are preserved."""
        request_data = {
            "image_url": {"url": "https://example.com/image.png"},
            "avatar_url": "https://example.com/avatar.jpg",
        }

        sanitized = CachingInterceptor.sanitize_request_data_for_logging(request_data)

        # Non-base64 URLs should be preserved
        assert sanitized["image_url"]["url"] == "https://example.com/image.png"
        assert sanitized["avatar_url"] == "https://example.com/avatar.jpg"
