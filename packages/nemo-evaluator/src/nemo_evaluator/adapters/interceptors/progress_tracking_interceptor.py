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

import os
import pathlib
import threading
import time
from typing import Annotated, Optional, final

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
            description="URL to post the number of processed samples to. Supports expansion of shell variables if present.",
        )
        progress_tracking_interval: Annotated[int, Field(gt=0)] = Field(
            default=1,
            description="How often (every how many samples) to send a progress information.",
        )
        progress_tracking_interval_seconds: Optional[
            Annotated[float | None, Field(gt=0)]
        ] = Field(
            default=None,
            description="How often (every N seconds) to send a progress information in addition to progress_tracking_interval.",
        )
        request_method: str = Field(
            default="PATCH",
            description="Request method to use for updating the evaluation progress.",
        )
        output_dir: Optional[str] = Field(
            default=None,
            description="Evaluation output directory. If provided, the progress tracking will be saved to a file in this directory.",
        )

    progress_tracking_url: Optional[str]
    progress_tracking_interval: int
    progress_filepath: Optional[pathlib.Path]
    request_method: str

    def __init__(self, params: Params):
        """
        Initialize the progress tracking interceptor.

        Args:
            params: Configuration parameters
        """
        self.progress_tracking_url = os.path.expandvars(params.progress_tracking_url)
        self.progress_tracking_interval = params.progress_tracking_interval
        self.request_method = params.request_method
        if params.output_dir is not None:
            output_dir = pathlib.Path(params.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.progress_filepath = output_dir / "progress"
        else:
            self.progress_filepath = None
        self._samples_processed = self._initialize_samples_processed()
        self._last_updated_samples_processed = self._samples_processed
        self._lock = threading.Lock()

        # Get logger for this interceptor with interceptor context
        self.logger = get_logger(self.__class__.__name__)

        # Optional update on timer
        self.progress_tracking_interval_seconds = (
            params.progress_tracking_interval_seconds
        )
        if self.progress_tracking_interval_seconds:
            self._timer_stopped = False
            self._update_on_timer_thread = threading.Thread(
                target=self._update_on_timer,
                kwargs={"interval_seconds": self.progress_tracking_interval_seconds},
                daemon=True,
            )
            self._update_on_timer_thread.start()

        self.logger.info(
            "Progress tracking interceptor initialized",
            progress_tracking_url=self.progress_tracking_url,
            progress_tracking_interval=self.progress_tracking_interval,
            progress_tracking_interval_seconds=self.progress_tracking_interval_seconds,
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

    def _send_progress(self, num_samples: int) -> requests.Response:
        self.logger.debug(
            "Sending progress to tracking server",
            url=self.progress_tracking_url,
            method=self.request_method,
            samples_processed=num_samples,
        )
        body = {"samples_processed": num_samples}
        try:
            resp = requests.request(
                self.request_method,
                self.progress_tracking_url,
                json=body,
            )
            if resp.status_code >= 200 and resp.status_code < 300:
                self.logger.debug(
                    "Progress sent successfully", samples_processed=num_samples
                )
            else:
                self.logger.warning(
                    "Failed to update job progress",
                    body=body,
                    status_code=resp.status_code,
                    response_text=resp.text,
                )
            return resp
        except requests.exceptions.RequestException as e:
            self.logger.error(
                "Failed to communicate with progress tracking server",
                error=str(e),
                samples_processed=num_samples,
            )

    def _update_on_timer(self, interval_seconds: float):
        """
        Sends an update on a timed interval if there has been a change since the last update.
        This is a blocking function that is expected to be executed in a thread.
        """
        assert interval_seconds > 0
        while True:
            time.sleep(interval_seconds)
            with self._lock:
                if self._timer_stopped:
                    return
                if self._last_updated_samples_processed == self._samples_processed:
                    continue
                curr_samples = self._samples_processed

            if self.progress_tracking_url is not None:
                self._send_progress(curr_samples)
            if self.progress_filepath is not None:
                self._write_progress(curr_samples)

            self.logger.info(
                "Progress milestone updated on time interval",
                samples_processed=curr_samples,
                interval=self.progress_tracking_interval,
            )
            with self._lock:
                self._last_updated_samples_processed = curr_samples

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
            with self._lock:
                self._last_updated_samples_processed = curr_samples

        return ar

    def post_eval_hook(self, context: AdapterGlobalContext) -> None:
        self.logger.info(
            "Post-eval hook executed", total_samples_processed=self._samples_processed
        )
        with self._lock:
            if self.progress_tracking_interval_seconds:
                self._timer_stopped = True
            if self._samples_processed == self._last_updated_samples_processed:
                return

        if self.progress_tracking_url is not None:
            self._send_progress(self._samples_processed)
        if self.progress_filepath is not None:
            self._write_progress(self._samples_processed)
