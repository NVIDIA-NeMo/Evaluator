# ---------------------------------------------------------------------------
# EFS — shared filesystem for workspace transfer between agent and verifier
# Fargate tasks.  The orchestrator creates per-session subdirectories under
# the access point root; both agent and verifier mount the same subdir.
# ---------------------------------------------------------------------------

resource "aws_efs_file_system" "workspace" {
  count = var.enable_efs ? 1 : 0

  encrypted        = true
  performance_mode = "generalPurpose"
  throughput_mode  = "elastic"

  lifecycle_policy {
    transition_to_ia = "AFTER_7_DAYS"
  }

  tags = merge(var.tags, { Name = "${var.project}-workspace-efs" })
}

resource "aws_efs_mount_target" "workspace" {
  count = var.enable_efs ? var.subnet_count : 0

  file_system_id  = aws_efs_file_system.workspace[0].id
  subnet_id       = aws_subnet.public[count.index].id
  security_groups = [aws_security_group.efs[0].id]
}

resource "aws_efs_access_point" "workspace" {
  count = var.enable_efs ? 1 : 0

  file_system_id = aws_efs_file_system.workspace[0].id

  posix_user {
    uid = 1000
    gid = 1000
  }

  root_directory {
    path = "/${var.project}/xfer"
    creation_info {
      owner_uid   = 1000
      owner_gid   = 1000
      permissions = "0755"
    }
  }

  tags = merge(var.tags, { Name = "${var.project}-workspace-ap" })
}

# ---------------------------------------------------------------------------
# Security group for EFS mount targets — NFS ingress from ECS tasks only.
# No egress rules needed: mount targets never initiate connections, and
# security groups are stateful so return traffic is auto-allowed.
# ---------------------------------------------------------------------------

resource "aws_security_group" "efs" {
  count = var.enable_efs ? 1 : 0

  name_prefix = "${var.project}-efs-"
  description = "EFS mount targets - NFS from ECS tasks"
  vpc_id      = aws_vpc.main.id

  ingress {
    from_port       = 2049
    to_port         = 2049
    protocol        = "tcp"
    security_groups = [aws_security_group.ecs_tasks.id]
    description     = "NFS from ECS tasks"
  }

  tags = merge(var.tags, { Name = "${var.project}-efs-sg" })

  lifecycle {
    create_before_destroy = true
  }
}
