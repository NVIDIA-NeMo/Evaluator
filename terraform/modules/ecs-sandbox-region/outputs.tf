output "ecs_cluster" {
  description = "ECS cluster name"
  value       = aws_ecs_cluster.main.name
}

output "task_definition" {
  description = "ECS base task definition family"
  value       = aws_ecs_task_definition.base.family
}

output "subnets" {
  description = "Public subnet IDs"
  value       = aws_subnet.public[*].id
}

output "security_groups" {
  description = "Security group IDs for ECS tasks"
  value       = [aws_security_group.ecs_tasks.id]
}

output "ecr_repository_url" {
  description = "ECR repository URL"
  value       = aws_ecr_repository.main.repository_url
}

output "s3_bucket" {
  description = "S3 staging bucket name"
  value       = aws_s3_bucket.staging.id
}

output "s3_bucket_arn" {
  description = "S3 staging bucket ARN"
  value       = aws_s3_bucket.staging.arn
}

output "log_group" {
  description = "CloudWatch log group name"
  value       = aws_cloudwatch_log_group.ecs.name
}

output "log_group_arn" {
  description = "CloudWatch log group ARN"
  value       = aws_cloudwatch_log_group.ecs.arn
}

output "ssh_tunnel_public_key_secret_arn" {
  description = "SSH public key secret ARN"
  value       = aws_secretsmanager_secret.ssh_pubkey.arn
}

output "ssh_tunnel_private_key_secret_arn" {
  description = "SSH private key secret ARN"
  value       = aws_secretsmanager_secret.ssh_privkey.arn
}

output "dockerhub_secret_arn" {
  description = "Docker Hub credentials secret ARN"
  value       = aws_secretsmanager_secret.dockerhub.arn
}

output "ssm_config_parameter_name" {
  description = "SSM Parameter Store path for NEL auto-discovery config"
  value       = aws_ssm_parameter.ecs_sandbox_config.name
}

output "vpc_id" {
  description = "VPC ID"
  value       = aws_vpc.main.id
}

output "efs_filesystem_id" {
  description = "EFS filesystem ID for workspace transfer (empty if EFS disabled)"
  value       = var.enable_efs ? aws_efs_file_system.workspace[0].id : ""
}

output "efs_access_point_id" {
  description = "EFS access point ID for workspace transfer (empty if EFS disabled)"
  value       = var.enable_efs ? aws_efs_access_point.workspace[0].id : ""
}
