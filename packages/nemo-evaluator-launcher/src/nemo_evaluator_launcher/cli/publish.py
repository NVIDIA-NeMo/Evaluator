# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
#
"""CLI wrapper for `nel publish`."""

import sys
from dataclasses import dataclass
from typing import Optional

from simple_parsing import field

from nemo_evaluator_launcher.api.functional import PublishError, publish_results
from nemo_evaluator_launcher.common.logging_utils import logger


@dataclass
class Cmd:
    """Publish an evaluation result to a HuggingFace dataset leaderboard.

    Examples:
      nel publish 097410193730c9e3 \\
          --hf-dataset-id Idavidrein/gpqa --hf-task-id diamond \\
          --metric gpqa.pass@2.symbolic_correct \\
          --hf-model-id nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 \\
          --dry-run
    """

    invocation_id: str = field(
        positional=True,
        help="Invocation or job ID of the evaluation run to publish.",
    )
    hf_dataset_id: str = field(
        alias=["--hf-dataset-id"],
        help="HuggingFace dataset repo id to publish to, as '<org>/<dataset>' "
        "(e.g. 'Idavidrein/gpqa').",
    )
    metric: str = field(
        alias=["--metric"],
        help="Score to extract from artifacts and publish. "
        "Provide either full path to the score in the results.yml "
        "(e.g. 'groups.gpqa.metrics.pass@2.scores.symbolic_correct') "
        "or use simplified syntax (e.g. 'gpqa.pass@2.symbolic_correct').",
    )
    hf_org: str = field(
        alias=["--hf-org"],
        help="HuggingFace org under which to create the traces Space "
        "(e.g. 'nvidia' or your HF username).",
    )
    hf_task_id: Optional[str] = field(
        default=None,
        alias=["--hf-task-id"],
        help="Task identifier within the dataset's eval.yaml (e.g. 'diamond'). "
        "Can be omitted when the dataset's eval.yaml defines exactly one task.",
    )
    hf_model_id: Optional[str] = field(
        default=None,
        alias=["--hf-model-id"],
        help="HuggingFace model repo to PR against (e.g. 'nvidia/model-name'). "
        "If omitted, inferred from the eval config's target.api_endpoint.model_id.",
    )
    overwrite: bool = field(
        default=False,
        alias=["--overwrite"],
        help="Replace an existing leaderboard entry for this model+dataset+task. "
        "If false (default) and an entry already exists, the command exits cleanly.",
    )
    dry_run: bool = field(
        default=False,
        alias=["-n", "--dry-run"],
        help="Render and print the leaderboard YAML entry without uploading "
        "traces or opening a PR.",
    )

    def execute(self) -> None:
        try:
            publish_results(
                invocation_id=self.invocation_id,
                hf_dataset_id=self.hf_dataset_id,
                score_spec=self.metric,
                hf_task_id=self.hf_task_id,
                hf_model_id=self.hf_model_id,
                hf_org=self.hf_org,
                overwrite=self.overwrite,
                dry_run=self.dry_run,
            )
        except PublishError as e:
            logger.error(f"publish failed: {e}")
            sys.exit(1)
