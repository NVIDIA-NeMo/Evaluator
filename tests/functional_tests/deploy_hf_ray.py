# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import logging

from nemo_eval.api import deploy

logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(
        description="Test evaluation with lm-eval-harness on HuggingFace model deployed on Ray"
    )
    parser.add_argument("--hf_model_id_path", type=str, help="HuggingFace model ID or local path to the model")
    parser.add_argument("--max_batch_size", type=int, help="Max batch size for the model")
    parser.add_argument("--port", type=int, help="Port")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    try:
        deploy(
            hf_model_id_path=args.hf_model_id_path,
            serving_backend="ray",
            model_name="hf_model",
            max_batch_size=args.max_batch_size,
            server_port=args.port,
            use_vllm_backend=False,
        )
    except Exception as e:
        logger.error(f"Deploy process encountered an error: {e}")
    logger.info("Deploy process terminated.")
