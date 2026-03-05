"""Wraps EvalEnvironment as a Gym resource server.

Fills nemo-gym's resources_servers/nemo_evaluator/ slot, making
any Evaluator benchmark available to ng_collect_rollouts.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from nemo_evaluator.environments.base import EvalEnvironment
from nemo_evaluator.environments.gym_protocol import (
    GymVerifyRequest,
    GymVerifyResponse,
    extract_assistant_text,
)

logger = logging.getLogger(__name__)


class GymHarness:
    def __init__(self, env: EvalEnvironment, eval_type: str | None = None, **kwargs: Any) -> None:
        self.env = env
        self.eval_type = eval_type or env.name

    def get_dataset(self) -> list[dict[str, Any]]:
        rows = []
        for idx in range(len(self.env)):
            seed = self.env.seed(idx)
            rows.append({
                "responses_create_params": {
                    "input": [{"role": "user", "content": seed.prompt}],
                },
                "expected_answer": seed.expected_answer,
                "uuid": f"{self.env.name}-{idx}",
                "eval_type": self.eval_type,
                "metadata": seed.metadata,
            })
        return rows

    def export_jsonl(self, output_dir: str | Path) -> Path:
        rows = self.get_dataset()
        path = Path(output_dir) / f"{self.eval_type}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        return path

    async def verify(self, text: str, expected: str, **meta: Any) -> tuple[float, str | None]:
        result = self.env.verify(text, expected, **meta)
        return result.reward, result.extracted_answer

    def as_fastapi_app(self) -> FastAPI:
        app = FastAPI(title=f"Gym Resource Server – {self.env.name}")
        harness = self

        @app.get("/health")
        async def health():
            return {"status": "ok", "benchmark": harness.env.name,
                    "dataset_size": len(harness.env)}

        @app.post("/seed_session")
        async def seed_session():
            return {}

        @app.post("/verify")
        async def verify(body: GymVerifyRequest):
            text = extract_assistant_text(body.response)
            reward, extracted = await harness.verify(text, body.expected_answer, **body.metadata)
            return GymVerifyResponse(
                responses_create_params=body.responses_create_params,
                response=body.response, reward=reward,
                expected_answer=body.expected_answer, extracted_answer=extracted)

        @app.get("/dataset_size")
        async def dataset_size():
            return {"size": len(harness.env)}

        return app


Harness = GymHarness
