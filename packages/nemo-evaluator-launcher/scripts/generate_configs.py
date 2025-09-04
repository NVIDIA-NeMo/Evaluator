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
