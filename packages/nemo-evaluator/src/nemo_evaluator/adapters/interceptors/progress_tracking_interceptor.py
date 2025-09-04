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

"""Progress tracking interceptor that tracks number of samples processed via webhook."""

import pathlib
import threading
from typing import Optional, final

import requests
from pydantic import Field

from nemo_evaluator.adapters.decorators import register_for_adapter
from nemo_evaluator.adapters.types import (
    AdapterGlobalContext,
    AdapterResponse,
    PostEvalHook,
    ResponseInterceptor,
)
from nemo_evaluator.logging import BaseLoggingParams, get_logger


@register_for_adapter(
    name="progress_tracking",
    description="Tracks number of samples processed via webhook",
)
@final
class ProgressTrackingInterceptor(ResponseInterceptor, PostEvalHook):
    """Progress tracking via external webhook."""

    class Params(BaseLoggingParams):
        """Configuration parameters for progress tracking interceptor."""

        progress_tracking_url: Optional[str] = Field(
            default="http://localhost:8000",
            description="URL to post the number of processed samples to.",
        )
        progress_tracking_interval: int = Field(
            default=1,
            description="How often (every how many samples) to send a progress information.",
        )
        output_dir: Optional[str] = Field(
            default=None,
            description="Evaluation output directory. If provided, the progress tracking will be saved to a file in this directory.",
        )

    progress_tracking_url: Optional[str]
    progress_tracking_interval: int
    progress_filepath: Optional[pathlib.Path]

    def __init__(self, params: Params):
        """
        Initialize the progress tracking interceptor.

        Args:
            params: Configuration parameters
        """
        self.progress_tracking_url = params.progress_tracking_url
        self.progress_tracking_interval = params.progress_tracking_interval
        if params.output_dir is not None:
            output_dir = pathlib.Path(params.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_filepath = output_dir / "progress"
        else:
            self.progress_filepath = None
        self._samples_processed = self._initialize_samples_processed()
        self._lock = threading.Lock()

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        self.logger.info(
            "Progress tracking interceptor initialized",
            progress_tracking_url=self.progress_tracking_url,
            progress_tracking_interval=self.progress_tracking_interval,
            output_dir=str(self.progress_filepath) if self.progress_filepath else None,
            initial_samples_processed=self._samples_processed,
        )

    def _initialize_samples_processed(self) -> int:
        if self.progress_filepath is not None and self.progress_filepath.exists():
            with open(self.progress_filepath, "r") as f:
                try:
                    return int(f.read())
                except ValueError:
                    self.logger.warning(
                        "Failed to read progress from file, starting from 0",
                        filepath=str(self.progress_filepath),
                    )
                    return 0
        return 0

    def _write_progress(self, num_samples: int):
        with self._lock:
            self.progress_filepath.write_text(str(num_samples))
            self.logger.debug(
                "Progress written to file",
                filepath=str(self.progress_filepath),
                samples_processed=num_samples,
            )

    def _send_progress(self, num_samples: int):
        self.logger.debug(
            "Sending progress to tracking server",
            url=self.progress_tracking_url,
            samples_processed=num_samples,
        )
        try:
            requests.post(
                self.progress_tracking_url,
                json={"samples_processed": num_samples},
            )
            self.logger.debug(
                "Progress sent successfully", samples_processed=num_samples
            )
        except requests.exceptions.RequestException as e:
            self.logger.error(
                "Failed to communicate with progress tracking server",
                error=str(e),
                samples_processed=num_samples,
            )

    @final
    def intercept_response(
        self, ar: AdapterResponse, context: AdapterGlobalContext
    ) -> AdapterResponse:
        curr_samples = 0
        with self._lock:
            self._samples_processed += 1
            curr_samples = self._samples_processed

        self.logger.debug(
            "Sample processed",
            samples_processed=curr_samples,
            interval=self.progress_tracking_interval,
        )

        if (curr_samples % self.progress_tracking_interval) == 0:
            if self.progress_tracking_url is not None:
                self._send_progress(curr_samples)
            if self.progress_filepath is not None:
                self._write_progress(curr_samples)

            self.logger.info(
                "Progress milestone reached",
                samples_processed=curr_samples,
                interval=self.progress_tracking_interval,
            )

        return ar

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        self.logger.info(
            "Post-eval hook executed", total_samples_processed=self._samples_processed
        )

        if self.progress_tracking_url is not None:
            self._send_progress(self._samples_processed)
        if self.progress_filepath is not None:
            self._write_progress(self._samples_processed)
