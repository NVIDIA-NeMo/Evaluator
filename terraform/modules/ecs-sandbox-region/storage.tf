# ---------------------------------------------------------------------------
# ECR repository — per-task Docker images pushed here by the orchestrator
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "main" {
  name                 = "${var.project}-${var.region}"
  image_tag_mutability = "MUTABLE"

  # Ephemeral eval env: terraform destroy wipes pushed images. Not for long-lived shared infra.
  force_delete = true

  image_scanning_configuration {
    scan_on_push = false
  }

  tags = var.tags
}

resource "aws_ecr_lifecycle_policy" "cleanup" {
  repository = aws_ecr_repository.main.name

  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep only 500 most recent images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 500
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# S3 bucket — staging area for file uploads/downloads and CodeBuild sources
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "staging" {
  bucket_prefix = "${var.project}-staging-"

  # Staging is short-lived: terraform destroy clears in-flight objects. 7-day lifecycle rule below caps churn.
  force_destroy = true

  tags = var.tags
}

resource "aws_s3_bucket_lifecycle_configuration" "staging_cleanup" {
  bucket = aws_s3_bucket.staging.id

  rule {
    id     = "expire-old-objects"
    status = "Enabled"

    filter {}

    expiration {
      days = 7
    }
  }
}

resource "aws_s3_bucket_public_access_block" "staging" {
  bucket = aws_s3_bucket.staging.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# ---------------------------------------------------------------------------
# CloudWatch log group
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "ecs" {
  name              = "/ecs/${var.project}/${var.region}"
  retention_in_days = 30

  tags = var.tags
}
