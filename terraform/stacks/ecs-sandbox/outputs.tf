# ---------------------------------------------------------------------------
# Shared outputs (global — IAM)
# ---------------------------------------------------------------------------

output "execution_role_arn" {
  description = "ECS task execution role ARN (shared across all regions)"
  value       = aws_iam_role.ecs_execution.arn
}

output "task_role_arn" {
  description = "ECS task role ARN (shared across all regions)"
  value       = aws_iam_role.ecs_task.arn
}

output "codebuild_service_role_arn" {
  description = "CodeBuild service role ARN (shared across all regions)"
  value       = aws_iam_role.codebuild.arn
}

output "orchestrator_policy_arn" {
  description = "Attach this policy to the IAM identity running NEL"
  value       = aws_iam_policy.orchestrator.arn
}

output "orchestrator_user_name" {
  description = "IAM user name for the NEL orchestrator"
  value       = aws_iam_user.orchestrator.name
}

output "orchestrator_access_key_id" {
  description = "AWS_ACCESS_KEY_ID for the NEL orchestrator"
  value       = aws_iam_access_key.orchestrator.id
}

output "orchestrator_secret_access_key" {
  description = "AWS_SECRET_ACCESS_KEY for the NEL orchestrator"
  value       = aws_iam_access_key.orchestrator.secret
  sensitive   = true
}

# ---------------------------------------------------------------------------
# Per-region outputs — maps keyed by region name
# ---------------------------------------------------------------------------

output "regions" {
  description = "List of deployed regions"
  value       = var.regions
}

output "vpc_ids" {
  description = "VPC ID per region"
  value       = { for region, mod in local.all_regions : region => mod.vpc_id }
}

output "ssm_config_parameter_names" {
  description = "SSM Parameter Store path for NEL auto-discovery config per region"
  value       = { for region, mod in local.all_regions : region => mod.ssm_config_parameter_name }
}
