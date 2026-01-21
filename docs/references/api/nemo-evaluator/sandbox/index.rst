``nemo_evaluator.sandbox``
======================================

Sandbox implementations used by evaluation harnesses that need a tmux-like interactive session.

This module is designed to keep dependencies **optional**:

- The ECS Fargate implementation only imports AWS SDKs (``boto3``/``botocore``) when actually used.
- Using the ECS sandbox also requires the AWS CLI (``aws``) and ``session-manager-plugin`` on the host.

Usage (ECS Fargate)
-------------------

Typical usage is:

- configure :class:`~nemo_evaluator.sandbox.ecs_fargate.EcsFargateConfig`
- :meth:`~nemo_evaluator.sandbox.ecs_fargate.EcsFargateSandbox.spin_up` a sandbox context
- create an interactive :class:`~nemo_evaluator.sandbox.base.NemoSandboxSession`

Example::

    from nemo_evaluator.sandbox import EcsFargateConfig, EcsFargateSandbox

    cfg = EcsFargateConfig(
        region="us-west-2",
        cluster="my-ecs-cluster",
        task_definition="my-task-def:1",
        container_name="eval",
        subnets=["subnet-abc"],
        security_groups=["sg-xyz"],
        s3_bucket="my-staging-bucket",
    )

    with EcsFargateSandbox.spin_up(
        cfg=cfg,
        task_id="task-001",
        trial_name="trial-0001",
        run_id="run-2026-01-12",
    ) as sandbox:
        session = sandbox.create_session("main")
        session.send_keys(["echo hello", "Enter"], block=True)
        print(session.capture_pane())

Prerequisites / Notes
---------------------

- The harness host must have **AWS CLI** and **session-manager-plugin** installed.
- If you use S3-based fallbacks (large uploads / long commands), configure ``s3_bucket``.

.. automodule:: nemo_evaluator.sandbox
    :members:
    :undoc-members:
    :member-order: bysource


