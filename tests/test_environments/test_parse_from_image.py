# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from nemo_evaluator.environments.harbor import _parse_from_image


def test_platform_flag_is_not_the_image(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM --platform=linux/amd64 ubuntu:22.04\n")
    assert _parse_from_image(dockerfile) == "ubuntu:22.04"

    dockerfile.write_text("FROM --platform linux/amd64 ubuntu:22.04 AS builder\n")
    assert _parse_from_image(dockerfile) == "ubuntu:22.04"


def test_plain_from_and_scratch_are_unchanged(tmp_path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text("FROM ubuntu:22.04 AS builder\n")
    assert _parse_from_image(dockerfile) == "ubuntu:22.04"

    dockerfile.write_text("FROM scratch\n")
    assert _parse_from_image(dockerfile) is None

    dockerfile.write_text("FROM $BASE\n")
    assert _parse_from_image(dockerfile) is None
