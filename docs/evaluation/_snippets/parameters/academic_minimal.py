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
#!/usr/bin/env python3
"""
Minimal configuration for academic benchmark evaluation.
"""

from nemo_evaluator.api.api_dataclasses import ConfigParams

# [snippet-start]
# Minimal configuration for academic benchmark evaluation
params = ConfigParams(
    temperature=0.01,  # Near-deterministic (0.0 not supported by all endpoints)
    top_p=1.0,  # No nucleus sampling
    max_new_tokens=256,  # Sufficient for most academic tasks
    limit_samples=100,  # Remove for full dataset
    parallelism=4,  # Adjust based on endpoint capacity
)
# [snippet-end]
