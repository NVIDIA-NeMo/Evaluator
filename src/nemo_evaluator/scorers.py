# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BYOB scorer adapters for reusable Metric implementations."""

from __future__ import annotations

from nemo_evaluator.environments.custom import scorer
from nemo_evaluator.metrics import ExactMatchMetric

__all__ = ["ExactMatchScorer"]


ExactMatchScorer = scorer(ExactMatchMetric)
