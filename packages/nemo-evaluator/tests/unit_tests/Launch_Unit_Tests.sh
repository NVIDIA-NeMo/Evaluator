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

export CUDA_VISIBLE_DEVICES=""

SCRIPT_DIR=$(dirname "$0")
PROJECT_DIR=$SCRIPT_DIR/../../
cd $PROJECT_DIR

coverage run \
    --data-file=.coverage.unit_tests \
    --source=src/ \
    -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -m "not pleasefixme" \
   tests/unit_tests
coverage combine -q