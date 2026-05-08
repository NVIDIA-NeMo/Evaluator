variable "region" {
  description = "AWS region for this regional deployment"
  type        = string
}

variable "project" {
  description = "Project name used as prefix for all resources"
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR block for the VPC in this region"
  type        = string
}

variable "subnet_count" {
  description = "Number of public subnets (one per AZ)"
  type        = number
  default     = 2
}

variable "ssh_tunnel_sshd_port" {
  description = "SSH sidecar port. Must match NEL DEFAULT_SSHD_PORT; avoid 2222 (TB2 QEMU hostfwd collision)."
  type        = number
  default     = 52222
}

variable "exec_server_port" {
  description = "Per-task exec server port. Must match NEL DEFAULT_EXEC_SERVER_PORT; avoid 5000 (TB2 hf-model-inference Flask collision)."
  type        = number
  default     = 19542
}

variable "ecs_task_cpu" {
  description = "Default CPU units for ECS tasks (1024 = 1 vCPU)"
  type        = number
}

variable "ecs_task_memory" {
  description = "Default memory (MiB) for ECS tasks"
  type        = number
}

variable "execution_role_arn" {
  description = "ARN of the shared ECS task execution IAM role"
  type        = string
}

variable "task_role_arn" {
  description = "ARN of the shared ECS task IAM role"
  type        = string
}

variable "ssh_public_key" {
  description = "SSH public key (OpenSSH format) for the tunnel sidecar"
  type        = string
}

variable "ssh_private_key" {
  description = "SSH private key (OpenSSH format) for the orchestrator"
  type        = string
  sensitive   = true
}

variable "dockerhub_username" {
  description = "Docker Hub username (optional, avoids pull rate limits)"
  type        = string
  default     = ""
}

variable "dockerhub_token" {
  description = "Docker Hub access token (optional, avoids pull rate limits)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "codebuild_service_role_arn" {
  description = "ARN of the shared CodeBuild service IAM role"
  type        = string
}

variable "enable_efs" {
  description = "Create an EFS filesystem for workspace transfer between agent/verifier tasks"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
}
