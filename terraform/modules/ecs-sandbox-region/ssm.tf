# ---------------------------------------------------------------------------
# SSM Parameter Store — ECS sandbox config for auto-discovery by NEL.
# NEL reads /{project}/ecs-sandbox/config in the target region to populate
# subnets, roles, SSH keys, EFS, etc. without manual copy-paste into YAML.
# ---------------------------------------------------------------------------

resource "aws_ssm_parameter" "ecs_sandbox_config" {
  name = "/${var.project}/ecs-sandbox/config"
  type = "String"
  tier = "Standard"

  description = "NEL ECS sandbox auto-discovery config (JSON) - managed by Terraform"

  value = jsonencode({
    cluster                = aws_ecs_cluster.main.name
    subnets                = aws_subnet.public[*].id
    security_groups        = [aws_security_group.ecs_tasks.id]
    assign_public_ip       = true
    execution_role_arn     = var.execution_role_arn
    task_role_arn          = var.task_role_arn
    log_group              = aws_cloudwatch_log_group.ecs.name
    s3_bucket              = aws_s3_bucket.staging.id
    ecr_repository         = aws_ecr_repository.main.repository_url
    codebuild_service_role = var.codebuild_service_role_arn
    dockerhub_secret_arn   = aws_secretsmanager_secret.dockerhub.arn
    efs_filesystem_id      = var.enable_efs ? aws_efs_file_system.workspace[0].id : ""
    efs_access_point_id    = var.enable_efs ? aws_efs_access_point.workspace[0].id : ""
    ssh_sidecar = {
      sshd_port              = var.ssh_tunnel_sshd_port
      exec_server_port       = var.exec_server_port
      public_key_secret_arn  = aws_secretsmanager_secret.ssh_pubkey.arn
      private_key_secret_arn = aws_secretsmanager_secret.ssh_privkey.arn
    }
  })

  tags = var.tags
}
