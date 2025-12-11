(how-to-switch-executors)=
# Switch Executor

In this guide you will learn how to switch from one executor to another.
For the purpose of this exercise we will use `local` and `slurm` executors with `vllm` model deployment.

:::{tip}
Learn more about the {ref}`execution-backend` concept and the {ref}`executors-overview` overview for details on available executors and their configuration.
:::

## Before You Start

Ensure you have:

- NeMo Evaluator Launcher config that you would like to run. You can use the config shown below, choose one of our [example configs](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/packages/nemo-evaluator-launcher/examples) or prepare your own configuration.
- NeMo Evaluator Launcher installed in your environment.
- Access to a Slurm cluster (with appropriate partitions/queues)
- [Pyxis SPANK plugin](https://github.com/NVIDIA/pyxis) installed on the cluster

## Starting Point: Config for Modifying


We will use the following config as our starting point:

```yaml
defaults:
  - execution: local
  - deployment: vllm
  - _self_

# set required execution arguments
execution:
  output_dir: local_results

deployment:
  checkpoint_path: null
  hf_model_handle: microsoft/Phi-4-mini-instruct
  served_model_name: microsoft/Phi-4-mini-instruct
  tensor_parallel_size: 1
  data_parallel_size: 1

evaluation:
  tasks:
    - name: ifeval  # chat benchmark will automatically use v1/chat/completions endpoint
    - name: gsm8k   # completions benchmark will automatically use v1/completions endpoint
```

This config will run deployment of Phi-4-mini-instruct with vLLM and evaluation on IFEval and GSM8k benchmarks.
The workflow is executed locally on a machine when you launch it.

## Modify the Config

In order to permanently switch to a different execution backend, you can modify the execution section of your config:

```yaml
defaults:
  - execution: local  # old executor: local
  - deployment: vllm
  - _self_

execution:
  output_dir: local_results   # path on your local machine
```

with a different one:

```yaml
defaults:
  - execution: slurm/default  # new executor: slurm
  - deployment: vllm
  - _self_

execution:
  hostname: my-cluster.login.com          # SLURM headnode (login) hostname
  account: my_account                     # SLURM account allocation
  output_dir: /absolute/path/on/remote    # ABSOLUTE path accessible to SLURM compute nodes
```

This will allow you to run the same deployment and evaluation workflow on a remote Slurm cluster.

## Dynamically switch executor with CLI overrides

You can also specify a different execution backend at runtime to dynamically switch from one executor to another:

```bash
export CLUSTER_HOSTNAME=my-cluster.login.com  # SLURM headnode (login) hostname
export ACCOUNT=my_account                     # SLURM account allocation
export OUT_DIR=/absolute/path/on/remote       # ABSOLUTE path accessible to SLURM compute nodes

nel run --config local_config.yaml \
  -o execution=slurm/default \
  -o execution.hostname=$CLUSTER_HOSTNAME \
  -o execution.account=$ACCOUNT \
  -o execution.output_dir=$OUT_DIR
```

This also allows you to easily switch from one Slurm cluster to another.
