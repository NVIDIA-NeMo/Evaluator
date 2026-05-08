variable "regions" {
  description = "List of AWS regions to deploy ECS sandbox infrastructure in"
  type        = list(string)
  default = [
    "us-east-1",
    "us-east-2",
    "us-west-1",
    "us-west-2",
    "ca-central-1",
    "eu-west-1",
    "eu-west-2",
    "eu-west-3",
    "eu-central-1",
    "eu-north-1",
    "sa-east-1",
  ]
}

variable "project" {
  description = "Project name used as prefix for all resources"
  type        = string
  default     = "nel-sandbox"
}

variable "vpc_base_cidr" {
  description = "Base CIDR block (/8) from which per-region /16 VPCs are derived via cidrsubnet"
  type        = string
  default     = "10.0.0.0/8"
}

variable "subnet_count" {
  description = "Number of public subnets per region (one per AZ)"
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
  default     = 4096
}

variable "ecs_task_memory" {
  description = "Default memory (MiB) for ECS tasks"
  type        = number
  default     = 16384
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

variable "enable_efs" {
  description = "Create an EFS filesystem per region for workspace transfer between agent/verifier tasks"
  type        = bool
  default     = true
}

variable "tags" {
  description = "Tags to apply to all resources"
  type        = map(string)
  default = {
    Project   = "nel-sandbox"
    ManagedBy = "terraform"
  }
}
