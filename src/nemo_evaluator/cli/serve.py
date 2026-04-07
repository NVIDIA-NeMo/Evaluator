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

import click


@click.command("serve")
@click.option("--benchmark", "-b", required=True)
@click.option("--host", default="0.0.0.0")
@click.option("--port", "-p", default=9090, type=int)
@click.option("--gym-compat", is_flag=True, help="Accept NeMoGymResponse in /verify")
@click.option("--export-data", type=click.Path(), default=None, help="Export JSONL for ng_collect_rollouts")
def serve_cmd(benchmark, host, port, gym_compat, export_data):
    """Serve a benchmark as HTTP environment."""
    from nemo_evaluator.environments.registry import get_environment
    from nemo_evaluator.serving.app import generate_app

    env = get_environment(benchmark)
    app = generate_app(env, gym_compat=gym_compat, data_export_dir=export_data)

    mode = "gym-compat" if gym_compat else "evaluator"
    click.echo(f"{env.name} on http://{host}:{port} ({mode})")
    click.echo(f"  size={len(env)}  endpoints: /seed_session /verify /health")

    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")
