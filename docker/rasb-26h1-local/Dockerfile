# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NEMO_EVALUATOR_RASB_ROOT=/rasb-26h1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends docker.io ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY packages/nemo-evaluator /tmp/nemo-evaluator
RUN pip install --no-cache-dir /tmp/nemo-evaluator

COPY rasb-26h1 /rasb-26h1
RUN pip install --no-cache-dir -r /rasb-26h1/requirements.txt

WORKDIR /rasb-26h1
