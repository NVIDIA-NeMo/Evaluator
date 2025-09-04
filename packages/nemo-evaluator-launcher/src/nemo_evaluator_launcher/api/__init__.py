"""nv-eval-platform public API package.

Exposes main API entry points and types for running evaluations and querying status.
"""

from nemo_evaluator_launcher.api.functional import get_status, get_tasks_list, run_eval
from nemo_evaluator_launcher.api.types import RunConfig
