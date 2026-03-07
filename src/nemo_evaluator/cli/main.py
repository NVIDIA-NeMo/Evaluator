"""CLI entrypoint: nel {eval, list, report, serve, validate, regression}."""

import click

from nemo_evaluator.cli.eval_cmd import eval_cmd
from nemo_evaluator.cli.list_cmd import list_cmd
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


if __name__ == "__main__":
    cli()
