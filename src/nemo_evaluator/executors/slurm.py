"""SlurmExecutor: generates and submits SLURM jobs for model + eval."""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

from nemo_evaluator.executors.base import RunConfig

logger = logging.getLogger(__name__)

_SBATCH_TEMPLATE = """\
#!/bin/bash
#SBATCH --job-name=nel-{job_name}
#SBATCH --output={output_dir}/slurm-%j.out
#SBATCH --error={output_dir}/slurm-%j.err
#SBATCH --time={time}
#SBATCH --nodes={nodes}
{gpu_line}
{partition_line}

set -euo pipefail

{deploy_section}

# Wait for model server
echo "Waiting for model server at $MODEL_URL..."
for i in $(seq 1 120); do
    if curl -sf "$MODEL_URL{health_path}" > /dev/null 2>&1; then
        echo "Model server ready."
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server died during startup."
        exit 1
    fi
    sleep 5
done

# Run evaluation
{nel_command}

# Cleanup
if [ -n "${{SERVER_PID:-}}" ]; then
    kill $SERVER_PID 2>/dev/null || true
fi
"""

_DEPLOY_DOCKER = """\
# Start model server via Docker
docker run --rm --gpus {gpus} -d \\
    --name nel-server-$SLURM_JOB_ID \\
    -p {port}:{port} \\
    {extra_env} \\
    {image} {extra_args} &
SERVER_PID=$!
MODEL_URL="http://localhost:{port}/v1"
MODEL_ID="{model_id}"
"""

class SlurmExecutor:
    """Generates SLURM sbatch script, submits it, returns job ID."""

    async def execute(self, config: RunConfig) -> dict[str, Any]:
        os.makedirs(config.output_dir, exist_ok=True)
        script = self._build_script(config)

        script_path = os.path.join(config.output_dir, "nel_eval.sbatch")
        with open(script_path, "w") as f:
            f.write(script)

        logger.info("Generated SLURM script: %s", script_path)

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr}")

        job_id = result.stdout.strip().split()[-1]
        logger.info("Submitted SLURM job: %s", job_id)

        return {
            "status": "submitted",
            "slurm_job_id": job_id,
            "script_path": script_path,
            "output_dir": config.output_dir,
        }

    def _build_script(self, config: RunConfig) -> str:
        deploy = config.deploy
        gpus = config.slurm_gpus or (deploy.gpus if deploy else 1)

        gpu_line = f"#SBATCH --gres=gpu:{gpus}" if gpus else ""
        partition_line = f"#SBATCH --partition={config.slurm_partition}" if config.slurm_partition else ""

        if deploy and deploy.type != "api":
            deploy_section = _DEPLOY_DOCKER.format(
                gpus=f'"device={",".join(str(i) for i in range(gpus))}"',
                port=deploy.port,
                image=deploy.image or "",
                model_id=deploy.model or config.model_id or "",
                extra_env=" ".join(f"-e {k}={v}" for k, v in deploy.extra_env.items()),
                extra_args=" ".join(deploy.extra_args),
            )
            health_path = deploy.health_path
        else:
            deploy_section = (
                f'MODEL_URL="{config.model_url}"\n'
                f'MODEL_ID="{config.model_id}"\n'
                'SERVER_PID=""'
            )
            health_path = "/v1/health/ready"

        nel_cmd = (
            f"nel run --env {config.env} "
            f'--model-url "$MODEL_URL" --model-id "$MODEL_ID" '
            f"--repeats {config.repeats} "
            f"-o {config.output_dir}"
        )
        if config.api_key:
            nel_cmd += f" --api-key {config.api_key}"
        if config.max_problems:
            nel_cmd += f" --max-problems {config.max_problems}"

        return _SBATCH_TEMPLATE.format(
            job_name=config.env.replace("://", "_").replace("/", "_"),
            output_dir=config.output_dir,
            time=config.slurm_time,
            nodes=config.slurm_nodes,
            gpu_line=gpu_line,
            partition_line=partition_line,
            deploy_section=deploy_section,
            health_path=health_path,
            nel_command=nel_cmd,
        )
