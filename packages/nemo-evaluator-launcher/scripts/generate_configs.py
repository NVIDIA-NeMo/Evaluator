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

#!/usr/bin/env python3
"""Generates bunch of up-to-date configs for your inspection."""

from omegaconf import OmegaConf

from nemo_evaluator_launcher.api import RunConfig


def main() -> None:
    for depl in ["vllm", "sglang", "none"]:
        for exec in [
            "local",
            "slurm/eos",
            "slurm/cw",
            "slurm/oci-ord",
        ]:
            cfg: RunConfig = RunConfig.from_hydra(
                hydra_overrides=[f"deployment={depl}", f"execution={exec}"]
            )
            yaml_str = OmegaConf.to_yaml(cfg)
            with open(f"{depl}_{exec.replace('/', '-')}.yaml", "w") as f:
                f.write(yaml_str)


if __name__ == "__main__":
    main()
