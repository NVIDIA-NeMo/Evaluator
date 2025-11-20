# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import logging
import subprocess
import time

from nemo_evaluator.api import check_endpoint, evaluate

logger = logging.getLogger(__name__)


def wait_and_evaluate(target_cfg, eval_cfg, serving_backend="pytriton"):
    server_ready = check_endpoint(
        endpoint_url=target_cfg.api_endpoint.url,
        endpoint_type=target_cfg.api_endpoint.type,
        model_name=target_cfg.api_endpoint.model_id,
    )
    if not server_ready:
        raise RuntimeError(
            "Server is not ready to accept requests. Check the deployment logs for errors."
        )

    result = evaluate(target_cfg=target_cfg, eval_cfg=eval_cfg)

    # Shutdown Ray server after evaluation if using Ray backend, since it does not shutdown automatically like Triton.
    if serving_backend == "ray":
        logger.info("Evaluation completed. Shutting down Ray server...")

        # Try to shutdown Ray using ray CLI
        subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)
        logger.info("Ray server shutdown command sent.")
        time.sleep(5)  # Give some time for graceful shutdown

    return result
