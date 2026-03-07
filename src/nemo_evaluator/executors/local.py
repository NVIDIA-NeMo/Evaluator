"""LocalExecutor: runs eval in-process, optionally deploys model via Docker."""
from __future__ import annotations

import logging
from typing import Any

from nemo_evaluator.executors.base import RunConfig

logger = logging.getLogger(__name__)


class LocalExecutor:
    async def execute(self, config: RunConfig) -> dict[str, Any]:
        from nemo_evaluator.environments.registry import get_environment
        from nemo_evaluator.observability.progress import ConsoleProgress
        from nemo_evaluator.runner.artifacts import write_all
        from nemo_evaluator.runner.deployment import get_deployment
        from nemo_evaluator.runner.eval_loop import run_evaluation
        from nemo_evaluator.runner.model_client import ModelClient
        from nemo_evaluator.runner.solver import ChatSolver

        deployment = None
        judge_deployment = None
        model_url = config.model_url
        model_id = config.model_id
        judge_client = None

        try:
            if config.deploy and config.deploy.type != "api":
                deployment = get_deployment(config.deploy)
                model_url = deployment.start()
                model_id = config.deploy.model or config.model_id
                logger.info("Model deployed at %s", model_url)
            elif not model_url:
                raise ValueError("Either --model-url or --deploy is required")

            if config.judge:
                if config.judge.deploy and config.judge.deploy.type != "api":
                    judge_deployment = get_deployment(config.judge.deploy)
                    judge_url = judge_deployment.start()
                    judge_id = config.judge.deploy.model or config.judge.model_id
                else:
                    judge_url = config.judge.model_url
                    judge_id = config.judge.model_id

                if judge_url and judge_id:
                    judge_client = ModelClient(
                        base_url=judge_url, model=judge_id,
                        api_key=config.judge.api_key or config.api_key,
                        temperature=0.0, max_tokens=config.max_tokens,
                    )

            env = get_environment(config.env, num_examples=config.max_problems)

            client = ModelClient(
                base_url=model_url, model=model_id, api_key=config.api_key,
                temperature=config.temperature, max_tokens=config.max_tokens,
                top_p=config.top_p, seed=config.seed, cache_dir=config.cache_dir,
            )
            solver = ChatSolver(client, system_prompt=config.system_prompt)

            run_config = {
                "benchmark": getattr(env, "name", config.env),
                "model": model_id, "base_url": model_url,
                "repeats": config.repeats, "max_problems": config.max_problems,
            }

            bundle = await run_evaluation(
                env, solver, n_repeats=config.repeats,
                max_problems=config.max_problems,
                config=run_config, progress=ConsoleProgress(),
                judge_client=judge_client,
            )

            write_all(bundle, config.output_dir)
            return bundle

        finally:
            if deployment:
                deployment.stop()
            if judge_deployment:
                judge_deployment.stop()
