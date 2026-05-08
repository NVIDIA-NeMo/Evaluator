# ---------------------------------------------------------------------------
# VPC — public subnets with IGW (required for assignPublicIp on Fargate)
# ---------------------------------------------------------------------------

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  public_subnet_cidrs = [
    for i in range(var.subnet_count) : cidrsubnet(var.vpc_cidr, 4, i + 1)
  ]
}

resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, { Name = "${var.project}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(var.tags, { Name = "${var.project}-igw" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.tags, { Name = "${var.project}-public-rt" })
}

resource "aws_subnet" "public" {
  count                   = var.subnet_count
  vpc_id                  = aws_vpc.main.id
  cidr_block              = local.public_subnet_cidrs[count.index]
  availability_zone       = data.aws_availability_zones.available.names[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, {
    Name = "${var.project}-public-${count.index}"
  })
}

resource "aws_route_table_association" "public" {
  count          = length(aws_subnet.public)
  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

# ---------------------------------------------------------------------------
# Security group — outbound all, inbound only SSH tunnel sidecar port
# ---------------------------------------------------------------------------

resource "aws_security_group" "ecs_tasks" {
  name_prefix = "${var.project}-ecs-"
  description = "ECS Fargate tasks"
  vpc_id      = aws_vpc.main.id

  # Outbound: allow everything (ECR pulls, S3, internet)
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Allow all outbound"
  }

  # Inbound: only the sshd port for the reverse tunnel
  ingress {
    from_port   = var.ssh_tunnel_sshd_port
    to_port     = var.ssh_tunnel_sshd_port
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH tunnel sidecar inbound"
  }

  tags = merge(var.tags, { Name = "${var.project}-ecs-sg" })

  lifecycle {
    create_before_destroy = true
  }
}
