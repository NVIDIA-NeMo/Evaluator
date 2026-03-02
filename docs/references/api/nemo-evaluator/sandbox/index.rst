``nemo_evaluator.sandbox``
======================================

Sandbox implementations used by evaluation harnesses that need isolated
container environments for command execution, file transfer, and agent hosting.

This module is designed to keep dependencies **optional**:

- The ECS Fargate implementation only imports AWS SDKs (``boto3``/``botocore``) when actually used.
- Transport is SSH-based; no AWS CLI or session-manager-plugin required on the host.

Sandbox Protocol
----------------

All sandbox backends implement the :class:`~nemo_evaluator.sandbox.base.Sandbox`
protocol, so harnesses can be written backend-agnostically:

- ``start()`` / ``stop()`` — lifecycle
- ``exec(command)`` — run a shell command
- ``upload(local_path, remote_path)`` / ``download(remote_path, local_path)`` — file transfer
- ``is_running`` — health check
- Context manager (``with sandbox: ...``) for automatic cleanup

Usage (ECS Fargate — exec-server mode)
--------------------------------------

For harnesses that drive execution from the orchestrator (e.g. terminal-bench)::

    from nemo_evaluator.sandbox import EcsFargateConfig, EcsFargateSandbox, SshSidecarConfig

    cfg = EcsFargateConfig(
        region="us-west-2",
        cluster="my-ecs-cluster",
        subnets=["subnet-abc"],
        security_groups=["sg-xyz"],
        image_template="123456789.dkr.ecr.us-west-2.amazonaws.com/my-repo:{task_id}",
        s3_bucket="my-staging-bucket",
        ssh_sidecar=SshSidecarConfig(
            public_key_secret_arn="arn:aws:secretsmanager:...:my-pubkey",
            private_key_secret_arn="arn:aws:secretsmanager:...:my-privkey",
            exec_server_port=19542,
        ),
    )

    with EcsFargateSandbox(cfg, task_id="task-001", run_id="run-001") as sandbox:
        sandbox.start()
        result = sandbox.exec("echo hello")
        print(result.stdout)

Usage (ECS Fargate — agent-server mode)
---------------------------------------

For harnesses that host an agent inside the container (e.g. openhands),
omit ``exec_server_port`` and configure reverse/forward tunnels::

    cfg = EcsFargateConfig(
        ...
        ssh_sidecar=SshSidecarConfig(
            public_key_secret_arn="arn:aws:secretsmanager:...:my-pubkey",
            private_key_secret_arn="arn:aws:secretsmanager:...:my-privkey",
            target_url_env="MODEL_URL",
        ),
    )

    with EcsFargateSandbox(cfg, task_id="task-001", run_id="run-001") as sandbox:
        sandbox.start()
        # The agent inside the container can now reach the model via the reverse tunnel.
        # The orchestrator can reach the agent's API via sandbox.local_port.

Prerequisites / Notes
---------------------

- SSH keys must be **pre-provisioned** in AWS Secrets Manager.
- If you use S3-based file staging (large uploads / downloads), configure ``s3_bucket``.
- Docker image building via AWS CodeBuild is available through :class:`~nemo_evaluator.sandbox.ecs_fargate.ImageBuilder`.

.. automodule:: nemo_evaluator.sandbox
    :members:
    :undoc-members:
    :member-order: bysource

