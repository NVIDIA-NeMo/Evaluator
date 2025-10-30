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
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

export CUDA_VISIBLE_DEVICES="0"
export HF_HOME="/home/TestData/HF_HOME"
export HF_DATASETS_OFFLINE="1"
export TRANSFORMERS_OFFLINE="1"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$SCRIPT_DIR/../../
cd $PROJECT_DIR

nemo2_ckpt_path="/home/TestData/nemo2_ckpt/llama-3_2-1b-instruct_v2.0"
model_name="megatron_model"
port=8886

python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint $nemo2_ckpt_path \
  --num_gpus 1 \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --model_id $model_name \
  --port $port &

deploy_pid=$!

coverage run \
    --data-file=.coverage.integration_tests \
    --source=src/ \
    -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -m "not pleasefixme" \
    -v -s \
   tests/integration_tests/nemo_fw/test_deployment.py

kill $deploy_pid
