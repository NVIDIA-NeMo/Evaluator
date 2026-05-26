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

# [snippet-start]
python \
  /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
  --nemo_checkpoint "meta-llama/Llama-3.1-8B" \
  --model_id "megatron_model" \
  --port 8080 \
  --num_gpus 4 \
  --num_replicas 2 \
  --tensor_model_parallel_size 2 \
  --pipeline_model_parallel_size 1 \
  --context_parallel_size 1
# [snippet-end]