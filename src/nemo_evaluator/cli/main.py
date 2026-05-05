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
"""CLI entrypoint: nel {eval, list, report, serve, validate, compare, gate, config, package, cache-sqsh}."""

import click

from nemo_evaluator.cli.cache_sqsh import cache_sqsh_cmd
from nemo_evaluator.cli.eval import eval_cmd
from nemo_evaluator.cli.gate import gate_cmd
from nemo_evaluator.cli.list import list_cmd
from nemo_evaluator.cli.package import package_cmd
from nemo_evaluator.cli.regression import compare_cmd
from nemo_evaluator.cli.report import report_cmd
from nemo_evaluator.cli.serve import serve_cmd
from nemo_evaluator.cli.settings import config_cmd
from nemo_evaluator.cli.validate import validate_cmd


@click.group()
@click.version_option(package_name="nemo-evaluator")
def cli():
    """NeMo Evaluator CLI."""


cli.add_command(eval_cmd, "eval")
cli.add_command(list_cmd, "list")
cli.add_command(report_cmd, "report")
cli.add_command(serve_cmd, "serve")
cli.add_command(validate_cmd, "validate")
cli.add_command(compare_cmd, "compare")
cli.add_command(gate_cmd, "gate")
cli.add_command(config_cmd, "config")
cli.add_command(package_cmd, "package")
cli.add_command(cache_sqsh_cmd, "cache-sqsh")


if __name__ == "__main__":
    cli()
