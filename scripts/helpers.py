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
import os
from pathlib import Path

from nemo_evaluator.api import check_endpoint, evaluate

logger = logging.getLogger(__name__)


def _write_eval_marker(success: bool):
    """Write marker files so the deploy wrapper knows eval finished.

    EVAL_DONE is always written (signals "eval is no longer running").
    EVAL_SUCCESS is only written on success.

    The deploy wrapper (RAY_DEPLOY_SCRIPT) polls for EVAL_DONE, then
    checks EVAL_SUCCESS to decide exit 0 (SUCCESS) or exit 1 (FAILED).
    """
    log_dir = os.environ.get("LOG_DIR", "/tmp")
    if success:
        Path(os.path.join(log_dir, "EVAL_SUCCESS")).touch()
        logger.info(f"Wrote EVAL_SUCCESS marker to {log_dir}")
    Path(os.path.join(log_dir, "EVAL_DONE")).touch()
    logger.info(f"Wrote EVAL_DONE marker to {log_dir}")


# [snippet-start]
def wait_and_evaluate(target_cfg, eval_cfg, serving_backend="pytriton"):
    try:
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

    except Exception:
        if serving_backend == "ray":
            logger.info("Evaluation failed. Writing failure marker.")
            _write_eval_marker(success=False)
        raise

    if serving_backend == "ray":
        logger.info("Evaluation completed successfully. Writing success marker.")
        _write_eval_marker(success=True)

    return result


# [snippet-end]
