# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tiny ATIF trajectory-aware retrieved context benchmark."""

from __future__ import annotations

from pathlib import Path

from nemo_evaluator.environments.custom import benchmark, scorer
from nemo_evaluator.metrics.retrieved_context import RetrievedContextPresenceMetric


DATASET = Path(__file__).parents[1] / "data" / "retrieved_context_presence.jsonl"
BENCHMARK_NAME = "retrieved-context-presence-poc"


retrieved_context_presence_poc = benchmark(
    name=BENCHMARK_NAME,
    dataset=str(DATASET),
    prompt="{question}",
    target_field="answer",
)(scorer(RetrievedContextPresenceMetric(expected_context="{{item.expected_context}}")))
