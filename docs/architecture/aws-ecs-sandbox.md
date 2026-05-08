# AWS ECS Fargate Sandbox — Reference Infrastructure

NEL's `ECSFargateSandbox` backend (see [Sandbox Orchestration](sandbox.md#ecsfargatesandbox)) runs each per-problem agent or code execution as an ECS Fargate task in your AWS account. The eval loop itself stays on whatever orchestrator host you choose (laptop, SLURM login node, Docker, K8s pod) — only the per-problem sandbox executes in ECS.

This page documents the **reference Terraform** that provisions the AWS-side infrastructure the `ecs_fargate` sandbox configuration expects. It is **not** a deployment guide for NEL itself.

The full implementation lives in [`terraform/`](https://github.com/NVIDIA-NeMo/Evaluator/tree/main/terraform); see [`terraform/README.md`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/terraform/README.md) for the operator-facing walkthrough.

## Where this fits

```{mermaid}
flowchart LR
    Orch["NEL orchestrator<br/>(local / SLURM / K8s / Docker)"] -->|"RunTask + SSH tunnels"| ECSTask["ECS Fargate task<br/>(per problem)"]
    Orch -->|reads| SSM["SSM Parameter<br/>/{ssm_project}/ecs-sandbox/config"]
    Orch -->|push image| ECR
    ECSTask -->|pull| ECR
    ECSTask -->|read secrets| Secrets["Secrets Manager"]
    ECSTask -->|stage files| S3["S3 staging bucket"]
    ECSTask -.->|optional| EFS["EFS workspace<br/>(agent ↔ verifier transfer)"]
```

When the YAML sets `region` and leaves `cluster` unset, the backend reads `/{ssm_project}/ecs-sandbox/config` from SSM and populates cluster, subnets, security groups, role ARNs, S3 bucket, ECR repository, EFS, and SSH-sidecar secret ARNs — anything you set in YAML still wins.

## What the Terraform creates

Per-region (one module instance per region in `var.regions`):

- **VPC** with public subnets across N AZs, internet gateway, route tables.
- **ECS Fargate cluster** with `containerInsights` and `execute-command` enabled.
- **Base task definition** (per-task tasks clone & override this).
- **ECR repository** with a 500-image lifecycle policy.
- **S3 staging bucket** with public access block and 7-day expiry.
- **CloudWatch log group** (30-day retention).
- **Secrets Manager** entries for the SSH-sidecar keypair and Docker Hub credentials.
- **SSM parameter** `/{project}/ecs-sandbox/config` with the JSON the sandbox backend reads.
- **Optional EFS workspace filesystem** for agent ↔ verifier file transfer.

Global (shared across all regions):

- **IAM roles**: ECS execution role, ECS task role, CodeBuild service role.
- **IAM user + access key + custom policy** for the orchestrator identity that submits `RunTask` and `StartBuild` calls.

## Prerequisites

- Terraform >= 1.5
- AWS credentials with permissions for VPC, ECS, ECR, S3, EFS, IAM, Secrets Manager, SSM, and CloudWatch Logs in every target region

## Quick start

```bash
cd terraform/stacks/ecs-sandbox
cp terraform.tfvars.example terraform.tfvars
# edit terraform.tfvars: set project, regions, tags
terraform init
terraform plan -out tfplan
terraform apply tfplan
```

A single-region apply takes 3–5 minutes (mostly EFS mount-target propagation when `enable_efs = true`).

## Wiring NEL to the provisioned cluster

Once apply completes:

1. Read the orchestrator credentials and feed them to your environment:

   ```bash
   terraform output orchestrator_access_key_id
   terraform output -raw orchestrator_secret_access_key
   ```

2. Reference the cluster in your benchmark config:

   ```yaml
   benchmarks:
     - name: harbor://swebench-verified@1.0
       solver:
         type: harbor
       sandbox:
         type: ecs_fargate
         region: us-west-2
         # ssm_project must match the Terraform `var.project` so NEL can
         # find /{ssm_project}/ecs-sandbox/config in SSM. Default is "harbor";
         # the example tfvars use "nel-sandbox", so set it here too.
         ssm_project: nel-sandbox
         # All other fields (cluster, subnets, sgs, ecr, ssh keys, EFS, …)
         # come from SSM auto-discovery because `cluster` is omitted.
   ```

   See `EcsFargateSandbox` in [`src/nemo_evaluator/config/sandboxes.py`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/src/nemo_evaluator/config/sandboxes.py) for the complete schema.

## State backend

Local state by default. To switch to a remote S3 backend, copy
[`terraform/stacks/ecs-sandbox/backend.tf.example`](https://github.com/NVIDIA-NeMo/Evaluator/blob/main/terraform/stacks/ecs-sandbox/backend.tf.example)
to `backend.tf`, supply your bucket and lock table, and run `terraform init -reconfigure`.

## Security

- The ECS-task security group's inbound `ssh_tunnel_sshd_port` is open to `0.0.0.0/0` for the MWE. Tighten it to the orchestrator's egress IP before any non-demo use.
- The orchestrator's IAM access key lives in Terraform state. Either keep state local and treated as a secret, or use an encrypted remote backend with strict access control.

## Tear down

```bash
terraform destroy
```

ECR, S3, and Secrets Manager are configured to clean up immediately.
