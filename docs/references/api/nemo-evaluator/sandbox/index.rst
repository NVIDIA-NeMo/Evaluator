``nemo_evaluator.sandbox``
======================================

Sandbox implementations used by evaluation harnesses that need a tmux-like interactive session.

This module is designed to keep dependencies **optional**:

- The ECS Fargate implementation only imports AWS SDKs (``boto3``/``botocore``) when actually used.
- Using the ECS sandbox also requires the AWS CLI (``aws``) and ``session-manager-plugin`` on the host.

.. automodule:: nemo_evaluator.sandbox
    :members:
    :undoc-members:
    :member-order: bysource


