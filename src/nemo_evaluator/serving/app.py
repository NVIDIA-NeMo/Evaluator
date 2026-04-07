# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from nemo_evaluator.environments.base import EvalEnvironment
from nemo_evaluator.environments.gym_protocol import extract_assistant_text

logger = logging.getLogger(__name__)


class SeedSessionRequest(BaseModel):
    idx: int = 0


class SeedSessionResponse(BaseModel):
    prompt: str = ""
    expected_answer: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    messages: list[dict[str, str]] | None = None
    system: str | None = None
    sandbox_spec: dict | None = None


class VerifyRequest(BaseModel):
    response: str
    expected: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifyResponse(BaseModel):
    reward: float
    extracted_answer: str | None = None
    scoring_details: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


async def _export_jsonl(env: EvalEnvironment, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for idx in range(len(env)):
            seed = await env.seed(idx)
            f.write(
                json.dumps(
                    {
                        "responses_create_params": {
                            "input": [{"role": "user", "content": seed.prompt}],
                        },
                        "expected_answer": seed.expected_answer,
                        "uuid": f"{env.name}-{idx}",
                        "metadata": seed.metadata,
                    }
                )
                + "\n"
            )
    logger.info("Exported %d rows to %s", len(env), path)


def generate_app(
    env: EvalEnvironment,
    gym_compat: bool = False,
    data_export_dir: str | None = None,
) -> FastAPI:
    app = FastAPI(title=f"NeMo Evaluator – {env.name}")
    ds_size = len(env)

    if data_export_dir:
        asyncio.run(_export_jsonl(env, Path(data_export_dir) / f"{env.name}.jsonl"))

    def _validate_idx(idx: int) -> None:
        if idx < 0 or idx >= ds_size:
            raise HTTPException(status_code=400, detail=f"idx={idx} out of range [0, {ds_size})")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "benchmark": env.name,
            "dataset_size": ds_size,
            "protocol": "gym" if gym_compat else "evaluator",
        }

    @app.post("/seed_session", response_model=SeedSessionResponse)
    async def seed_session(body: SeedSessionRequest = SeedSessionRequest()):
        _validate_idx(body.idx)
        sr = await env.seed(body.idx)
        spec_dict = None
        if sr.sandbox_spec is not None:
            from dataclasses import asdict

            spec_dict = asdict(sr.sandbox_spec)
        return SeedSessionResponse(
            prompt=sr.prompt,
            expected_answer=sr.expected_answer,
            metadata=sr.metadata,
            messages=sr.messages,
            system=sr.system,
            sandbox_spec=spec_dict,
        )

    @app.post("/verify")
    async def verify_unified(body: dict[str, Any] = {}):
        """Accept both evaluator format and gym format on the same endpoint.

        Evaluator format: {"response": str, "expected": str, "metadata": {}}
        Gym format: {"response": any, "expected_answer": str, "metadata": {}, ...}
        Detection: if "expected" key exists -> evaluator. Otherwise -> gym.
        """
        if "expected" in body:
            response_text = body.get("response", "")
            expected = body["expected"]
            meta = body.get("metadata", {})
            vr = await env.verify(response_text, expected, **meta)
            return VerifyResponse(
                reward=vr.reward,
                extracted_answer=vr.extracted_answer,
                scoring_details=vr.scoring_details,
                metadata=vr.metadata,
            )
        else:
            raw_response = body.get("response", "")
            text = extract_assistant_text(raw_response)
            expected = body.get("expected_answer", "") or body.get("expected", "")
            meta = body.get("metadata", {})
            vr = await env.verify(text, expected, **meta)
            resp = {
                "reward": vr.reward,
                "expected_answer": expected,
                "extracted_answer": vr.extracted_answer,
                "scoring_details": vr.scoring_details,
            }
            if "responses_create_params" in body:
                resp["responses_create_params"] = body["responses_create_params"]
            if "response" in body:
                resp["response"] = body["response"]
            return resp

    @app.get("/dataset_size")
    async def dataset_size():
        return {"size": ds_size}

    return app
