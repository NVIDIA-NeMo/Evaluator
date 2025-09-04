import json
from dataclasses import dataclass

from nemo_evaluator_launcher.api.functional import get_tasks_list


@dataclass
class Cmd:
    """List command configuration."""

    def execute(self) -> None:
        # TODO(dfridman): modify `get_tasks_list` to return a list of dicts in the first place
        data = get_tasks_list()
        headers = ["task", "endpoint_type", "harness", "container"]
        supported_benchmarks = []
        for task_data in data:
            assert len(task_data) == len(headers)
            supported_benchmarks.append(dict(zip(headers, task_data)))
        print(json.dumps(supported_benchmarks, indent=2))
