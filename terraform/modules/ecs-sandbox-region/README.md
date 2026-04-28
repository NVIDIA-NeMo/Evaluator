# Module: `ecs-sandbox-region`

Per-region building block for the AWS infrastructure consumed by NEL's `ECSFargateSandbox` backend (per-task sandbox executor). One instance of this module is provisioned per AWS region you want to use. The root stack ([`stacks/ecs-sandbox`](../../stacks/ecs-sandbox)) wires it up with an aliased provider and the shared IAM role ARNs.

## Resources created

| File | Resources |
|------|-----------|
| [`network.tf`](network.tf) | `aws_vpc`, `aws_internet_gateway`, `aws_subnet` (×N), `aws_route_table`(+ associations), `aws_security_group` for ECS tasks |
| [`ecs.tf`](ecs.tf) | `aws_ecs_cluster` (with execute-command + Container Insights), `aws_ecs_task_definition` (Fargate base) |
| [`storage.tf`](storage.tf) | `aws_ecr_repository` + lifecycle policy (keep 500 latest), `aws_s3_bucket` (staging, 7-day expiry, public-access block), `aws_cloudwatch_log_group` |
| [`secrets.tf`](secrets.tf) | `aws_secretsmanager_secret` × 3 (SSH pubkey, SSH privkey, Docker Hub creds) + initial versions |
| [`ssm.tf`](ssm.tf) | `aws_ssm_parameter` `/{project}/ecs-sandbox/config` (JSON; NEL auto-discovery) |
| [`efs.tf`](efs.tf) | Optional `aws_efs_file_system`, mount targets, access point, NFS security group |

## Inputs

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `region` | string | — | AWS region this module instance provisions resources in. Used in resource names and CloudWatch log group paths. |
| `project` | string | — | Name prefix for every resource. |
| `vpc_cidr` | string | — | `/16` CIDR block for the VPC. The root stack derives this with `cidrsubnet(var.vpc_base_cidr, 8, region_index)`. |
| `subnet_count` | number | `2` | Number of public subnets (one per AZ). |
| `ssh_tunnel_sshd_port` | number | — | Port the SSH-tunnel sidecar listens on (allowed inbound on the ECS task SG). |
| `ecs_task_cpu` | number | — | Fargate CPU units for the base task definition (`1024` = 1 vCPU). |
| `ecs_task_memory` | number | — | Fargate memory (MiB) for the base task definition. |
| `execution_role_arn` | string | — | ARN of the shared ECS task execution role (created in the root stack). |
| `task_role_arn` | string | — | ARN of the shared ECS task role (created in the root stack). |
| `ssh_public_key` | string | — | OpenSSH public key for the tunnel sidecar. |
| `ssh_private_key` | string (sensitive) | — | OpenSSH private key for the orchestrator. |
| `dockerhub_username` | string | `""` | Optional Docker Hub username. |
| `dockerhub_token` | string (sensitive) | `""` | Optional Docker Hub PAT. |
| `codebuild_service_role_arn` | string | — | ARN of the shared CodeBuild service role (created in the root stack). |
| `enable_efs` | bool | `false` | Create an EFS workspace filesystem in this region. |
| `tags` | `map(string)` | — | Tags applied to every resource. |

## Outputs

| Name | Description |
|------|-------------|
| `vpc_id` | VPC ID |
| `subnets` | Public subnet IDs |
| `security_groups` | List with the single ECS-task security group ID |
| `ecs_cluster` | ECS cluster name |
| `task_definition` | Base task definition family |
| `ecr_repository_url` | ECR repo URL |
| `s3_bucket` / `s3_bucket_arn` | S3 staging bucket |
| `log_group` / `log_group_arn` | CloudWatch log group |
| `ssh_tunnel_public_key_secret_arn` / `ssh_tunnel_private_key_secret_arn` | Secrets Manager ARNs for the SSH keypair |
| `dockerhub_secret_arn` | Secrets Manager ARN for Docker Hub credentials |
| `ssm_config_parameter_name` | Path to the JSON auto-discovery parameter |
| `efs_filesystem_id` / `efs_access_point_id` | EFS resource IDs (empty when `enable_efs = false`) |

## Provider requirements

Declared in [`versions.tf`](versions.tf):

- `hashicorp/aws ~> 5.0` with `configuration_aliases = [aws]` so the caller passes a region-specific provider via the module's `providers = { aws = aws.<region> }` block.

## Notes

- The security group allows **all egress** (needed for ECR pulls, S3, DockerHub, etc.) and inbound on `ssh_tunnel_sshd_port` from `0.0.0.0/0`. Tighten the inbound CIDR to the orchestrator's IP before using this for anything sensitive.
- The S3 bucket uses `bucket_prefix` so each apply gets a unique global name; `force_destroy = true` lets `terraform destroy` clean up even with objects present.
- Secrets use `recovery_window_in_days = 0` so they delete immediately on destroy. Adjust if you want AWS's default 7–30 day soft-delete window.
- The Docker Hub secret's `secret_string` is wrapped in `lifecycle { ignore_changes }` so subsequent applies don't overwrite a manually rotated credential. Rotate via `terraform apply -replace=...aws_secretsmanager_secret_version.dockerhub`.
