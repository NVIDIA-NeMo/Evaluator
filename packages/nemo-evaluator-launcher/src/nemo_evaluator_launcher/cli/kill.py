import json
from dataclasses import dataclass

from simple_parsing import field

from nemo_evaluator_launcher.api.functional import kill_job_or_invocation


@dataclass
class Cmd:
    """Kill command configuration."""

    id: str = field(
        positional=True,
        metadata={
            "help": "Job ID (e.g., aefc4819.0) or invocation ID (e.g., aefc4819) to kill"
        },
    )

    def execute(self):
        """Execute the kill command."""
        result = kill_job_or_invocation(self.id)
        # Output as JSON
        print(json.dumps(result, indent=2))
