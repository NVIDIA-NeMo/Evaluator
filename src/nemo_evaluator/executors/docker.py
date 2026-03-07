"""DockerExecutor: runs model server + eval container in Docker."""
from __future__ import annotations

import json
import logging
import os
import subprocess
from typing import Any


from nemo_evaluator.executors.base import RunConfig

logger = logging.getLogger(__name__)

_HEALTH_POLL_INTERVAL = 5.0


class DockerExecutor:
    """Orchestrates Docker containers: model server + eval runner."""

    async def execute(self, config: RunConfig) -> dict[str, Any]:
        from nemo_evaluator.runner.deployment import get_deployment

        deployment = None
        server_container = None

        try:
            # Step 1: Deploy model server
            if config.deploy and config.deploy.type != "api":
                deployment = get_deployment(config.deploy)
                model_url = deployment.start()
                model_id = config.deploy.model or config.model_id
                server_container = getattr(deployment, "_container_name", None)
            else:
                model_url = config.model_url
                model_id = config.model_id

            if not model_url or not model_id:
                raise ValueError("DockerExecutor requires model deployment or --model-url")

            # Step 2: Run eval in a container sharing the server's network
            eval_image = os.environ.get("NEL_EVAL_IMAGE", "nemo-evaluator:latest")
            nel_cmd = (
                f"nel run --env {config.env} "
                f"--model-url {model_url} --model-id {model_id} "
                f"--repeats {config.repeats} "
                f"-o /results"
            )
            if config.api_key:
                nel_cmd += f" --api-key {config.api_key}"
            if config.max_problems:
                nel_cmd += f" --max-problems {config.max_problems}"

            docker_args = ["docker", "run", "--rm", "-v", f"{os.path.abspath(config.output_dir)}:/results"]
            if server_container:
                docker_args += ["--network", f"container:{server_container}"]
            docker_args += [eval_image, "bash", "-c", nel_cmd]

            logger.info("Running eval container: %s", " ".join(docker_args))
            result = subprocess.run(docker_args, capture_output=True, text=True, timeout=7200)

            if result.returncode != 0:
                logger.error("Eval container failed: %s", result.stderr[-2000:])
                raise RuntimeError(f"Eval container exited {result.returncode}")

            # Step 3: Load results
            bundle_files = list(
                p for p in (
                    os.path.join(config.output_dir, f)
                    for f in os.listdir(config.output_dir)
                )
                if p.endswith(".json") and "eval-" in os.path.basename(p)
            )
            if bundle_files:
                with open(bundle_files[0]) as f:
                    return json.load(f)
            return {"status": "completed", "output_dir": config.output_dir}

        finally:
            if deployment:
                deployment.stop()
