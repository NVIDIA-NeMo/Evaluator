# ---------------------------------------------------------------------------
# Stable region → index map for VPC CIDR allocation.
# Each region always gets the same /16 regardless of which regions are active.
# ---------------------------------------------------------------------------

locals {
  region_cidr_index = {
    "us-east-1"    = 0
    "us-east-2"    = 1
    "us-west-1"    = 2
    "us-west-2"    = 3
    "ca-central-1" = 4
    "eu-west-1"    = 5
    "eu-west-2"    = 6
    "eu-west-3"    = 7
    "eu-central-1" = 8
    "eu-north-1"   = 9
    "sa-east-1"    = 10
  }
}

# ---------------------------------------------------------------------------
# SSH key pair — single key shared across all regions.  The key material
# is replicated into Secrets Manager in each region by the regional module.
# ---------------------------------------------------------------------------

resource "tls_private_key" "ssh_tunnel" {
  algorithm = "ED25519"
}

# ---------------------------------------------------------------------------
# Regional deployments — one conditional module instance per region.
# Modules are gated by var.regions so only requested regions are created.
# ---------------------------------------------------------------------------

module "us_east_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "us-east-1") ? 1 : 0
  providers = { aws = aws.us_east_1 }

  region                     = "us-east-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["us-east-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "us-east-1" })
}

module "us_east_2" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "us-east-2") ? 1 : 0
  providers = { aws = aws.us_east_2 }

  region                     = "us-east-2"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["us-east-2"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "us-east-2" })
}

module "us_west_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "us-west-1") ? 1 : 0
  providers = { aws = aws.us_west_1 }

  region                     = "us-west-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["us-west-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "us-west-1" })
}

module "us_west_2" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "us-west-2") ? 1 : 0
  providers = { aws = aws.us_west_2 }

  region                     = "us-west-2"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["us-west-2"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "us-west-2" })
}

module "ca_central_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "ca-central-1") ? 1 : 0
  providers = { aws = aws.ca_central_1 }

  region                     = "ca-central-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["ca-central-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "ca-central-1" })
}

module "eu_west_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "eu-west-1") ? 1 : 0
  providers = { aws = aws.eu_west_1 }

  region                     = "eu-west-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["eu-west-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "eu-west-1" })
}

module "eu_west_2" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "eu-west-2") ? 1 : 0
  providers = { aws = aws.eu_west_2 }

  region                     = "eu-west-2"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["eu-west-2"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "eu-west-2" })
}

module "eu_west_3" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "eu-west-3") ? 1 : 0
  providers = { aws = aws.eu_west_3 }

  region                     = "eu-west-3"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["eu-west-3"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "eu-west-3" })
}

module "eu_central_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "eu-central-1") ? 1 : 0
  providers = { aws = aws.eu_central_1 }

  region                     = "eu-central-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["eu-central-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "eu-central-1" })
}

module "eu_north_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "eu-north-1") ? 1 : 0
  providers = { aws = aws.eu_north_1 }

  region                     = "eu-north-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["eu-north-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "eu-north-1" })
}

module "sa_east_1" {
  source    = "../../modules/ecs-sandbox-region"
  count     = contains(var.regions, "sa-east-1") ? 1 : 0
  providers = { aws = aws.sa_east_1 }

  region                     = "sa-east-1"
  project                    = var.project
  vpc_cidr                   = cidrsubnet(var.vpc_base_cidr, 8, local.region_cidr_index["sa-east-1"])
  subnet_count               = var.subnet_count
  ssh_tunnel_sshd_port       = var.ssh_tunnel_sshd_port
  exec_server_port           = var.exec_server_port
  ecs_task_cpu               = var.ecs_task_cpu
  ecs_task_memory            = var.ecs_task_memory
  execution_role_arn         = aws_iam_role.ecs_execution.arn
  task_role_arn              = aws_iam_role.ecs_task.arn
  ssh_public_key             = trimspace(tls_private_key.ssh_tunnel.public_key_openssh)
  ssh_private_key            = tls_private_key.ssh_tunnel.private_key_openssh
  dockerhub_username         = var.dockerhub_username
  dockerhub_token            = var.dockerhub_token
  codebuild_service_role_arn = aws_iam_role.codebuild.arn
  enable_efs                 = var.enable_efs
  tags                       = merge(var.tags, { Region = "sa-east-1" })
}

# ---------------------------------------------------------------------------
# Aggregation — merge all active regional modules into a single map
# keyed by region for convenient output generation.
# ---------------------------------------------------------------------------

locals {
  all_regions = merge(
    length(module.us_east_1) > 0 ? { "us-east-1" = module.us_east_1[0] } : {},
    length(module.us_east_2) > 0 ? { "us-east-2" = module.us_east_2[0] } : {},
    length(module.us_west_1) > 0 ? { "us-west-1" = module.us_west_1[0] } : {},
    length(module.us_west_2) > 0 ? { "us-west-2" = module.us_west_2[0] } : {},
    length(module.ca_central_1) > 0 ? { "ca-central-1" = module.ca_central_1[0] } : {},
    length(module.eu_west_1) > 0 ? { "eu-west-1" = module.eu_west_1[0] } : {},
    length(module.eu_west_2) > 0 ? { "eu-west-2" = module.eu_west_2[0] } : {},
    length(module.eu_west_3) > 0 ? { "eu-west-3" = module.eu_west_3[0] } : {},
    length(module.eu_central_1) > 0 ? { "eu-central-1" = module.eu_central_1[0] } : {},
    length(module.eu_north_1) > 0 ? { "eu-north-1" = module.eu_north_1[0] } : {},
    length(module.sa_east_1) > 0 ? { "sa-east-1" = module.sa_east_1[0] } : {},
  )
}
