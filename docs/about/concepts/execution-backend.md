(execution-backend)=

# Execution Backend

NeMo Evaluator can be run in various environments: locally, on a cluster, or within NVIDIA Lepton. We refer to these environments as **execution backends** and we have corresponding **Executors** in NeMo Evaluator Launcher  that take care of evaluation orchestration in the designated backend. Each executor uses NVIDIA-built docker containers with pre-installed harnesses and the right container is automatically selected and run for you through the Launcher. 

## Executors

Different environments require  a bit different setup. This includes, among others,  submitting, launching and collecting results. Refer to the list below for executors available today. 

- **Local Executor**: orchestrates evaluations on a local machine using docker daemon.
- **SLURM executor**: orchestrates evaluations through a SLURM manager
- **Lepton Executor**: submit jobs via [Lepton AI](https://www.nvidia.com/en-us/data-center/dgx-cloud-lepton/)


:::{note} SLURM executor operates only on a SLURM cluster with pyxis SPANK plugin installed. Pyxis allows unprivileged cluster users to run containerized tasks through the `srun` command. Visit [pyxis github homepage](https://github.com/NVIDIA/pyxis) to learn more.
:::

Naturally, each executor might require additional configuration. For example, NeMo Evaluator Launcher needs information on partition, account and selected nodes on SLURM execution backend
