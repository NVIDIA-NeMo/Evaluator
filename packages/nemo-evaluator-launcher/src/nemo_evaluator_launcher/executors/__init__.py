# NOTE(dfridman): importing all executors will register them in the registry

from nemo_evaluator_launcher.executors.lepton.executor import LeptonExecutor
from nemo_evaluator_launcher.executors.local.executor import LocalExecutor
from nemo_evaluator_launcher.executors.slurm.executor import SlurmExecutor

__all__ = ["LeptonExecutor", "LocalExecutor", "SlurmExecutor"]
