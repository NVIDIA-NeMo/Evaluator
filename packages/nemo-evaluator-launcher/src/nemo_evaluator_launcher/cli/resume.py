"""CLI command for resuming evaluations by re-executing existing scripts."""

from dataclasses import dataclass

from simple_parsing import field


@dataclass
class Cmd:
    """Resume command configuration."""

    invocation_id: str = field(
        positional=True,
        metadata={"help": "Invocation ID to resume (supports partial IDs)"},
    )

    def execute(self) -> None:
        """Execute the resume command."""
        from nemo_evaluator_launcher.api.functional import resume_eval
        from nemo_evaluator_launcher.common.printing_utils import bold, cyan, green

        try:
            resumed_invocation_id = resume_eval(self.invocation_id)
            print(
                bold(cyan("To check status:"))
                + f" nemo-evaluator-launcher status {resumed_invocation_id}"
            )
            print(
                bold(cyan("To view job info:"))
                + f" nemo-evaluator-launcher info {resumed_invocation_id}"
            )
            print(green(bold(f"✓ Resumed invocation {resumed_invocation_id}")))
        except (ValueError, RuntimeError, FileNotFoundError) as e:
            print(f"Error: {e}")
            exit(1)
