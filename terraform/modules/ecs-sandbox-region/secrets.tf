# ---------------------------------------------------------------------------
# SSH key pair — stored in Secrets Manager (key material shared across regions,
# generated once in the root stack)
# ---------------------------------------------------------------------------

resource "aws_secretsmanager_secret" "ssh_pubkey" {
  name                    = "${var.project}-ssh-tunnel-pubkey-${var.region}"
  description             = "SSH public key for the ECS tunnel sidecar (authorized_keys)"
  recovery_window_in_days = 0 # allow immediate deletion

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "ssh_pubkey" {
  secret_id     = aws_secretsmanager_secret.ssh_pubkey.id
  secret_string = var.ssh_public_key
}

resource "aws_secretsmanager_secret" "ssh_privkey" {
  name                    = "${var.project}-ssh-tunnel-privkey-${var.region}"
  description             = "SSH private key for the orchestrator reverse tunnel"
  recovery_window_in_days = 0

  tags = var.tags
}

resource "aws_secretsmanager_secret_version" "ssh_privkey" {
  secret_id     = aws_secretsmanager_secret.ssh_privkey.id
  secret_string = var.ssh_private_key
}

# ---------------------------------------------------------------------------
# Docker Hub credentials (optional — avoids pull rate limits in CodeBuild)
#
# The secret always exists so that omitting dockerhub_username on a
# subsequent apply never triggers a destroy of existing credentials.
# The secret_version ignores changes to secret_string after initial
# creation; to rotate credentials, use:
#   terraform apply -replace="module.<region>[0].aws_secretsmanager_secret_version.dockerhub"
# ---------------------------------------------------------------------------

moved {
  from = aws_secretsmanager_secret.dockerhub[0]
  to   = aws_secretsmanager_secret.dockerhub
}

resource "aws_secretsmanager_secret" "dockerhub" {
  name                    = "${var.project}-dockerhub-${var.region}"
  description             = "Docker Hub credentials for CodeBuild image pulls"
  recovery_window_in_days = 0

  tags = var.tags
}

moved {
  from = aws_secretsmanager_secret_version.dockerhub[0]
  to   = aws_secretsmanager_secret_version.dockerhub
}

resource "aws_secretsmanager_secret_version" "dockerhub" {
  secret_id = aws_secretsmanager_secret.dockerhub.id
  secret_string = jsonencode({
    username = var.dockerhub_username
    password = var.dockerhub_token
  })

  lifecycle {
    ignore_changes = [secret_string]
  }
}
