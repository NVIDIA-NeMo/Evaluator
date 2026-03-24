"""CLI entrypoint: nel {eval, list, report, serve, validate, regression, config, package, cache-sqsh}."""

import click

from nemo_evaluator.cli.cache_sqsh import cache_sqsh_cmd
from nemo_evaluator.cli.settings import config_cmd
from nemo_evaluator.cli.eval import eval_cmd
from nemo_evaluator.cli.list import list_cmd
from nemo_evaluator.cli.package import package_cmd
from nemo_evaluator.cli.regression import regression_cmd
from nemo_evaluator.cli.report import report_cmd
from nemo_evaluator.cli.serve import serve_cmd
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
cli.add_command(regression_cmd, "regression")
cli.add_command(config_cmd, "config")
cli.add_command(package_cmd, "package")
cli.add_command(cache_sqsh_cmd, "cache-sqsh")


if __name__ == "__main__":
    cli()
