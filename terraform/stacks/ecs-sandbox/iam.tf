# ---------------------------------------------------------------------------
# ECS Task Execution Role — used by ECS agent to pull images, push logs,
# read secrets.  Shared across all regions (IAM is global).
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_execution" {
  name = "${var.project}-ecs-execution"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy_attachment" "ecs_execution_managed" {
  role       = aws_iam_role.ecs_execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "ecs_execution_extra" {
  name = "${var.project}-ecs-execution-extra"
  role = aws_iam_role.ecs_execution.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/ecs/${var.project}*"
      },
      {
        Sid      = "SecretsManager"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.project}-ssh-tunnel-*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# ECS Task Role — assumed by the running container.  Needs ECS Exec (SSM),
# S3 for file staging, and Secrets Manager for private key download.
# Shared across all regions.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "ecs_task" {
  name = "${var.project}-ecs-task"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "ecs-tasks.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "ecs_task_policy" {
  name = "${var.project}-ecs-task-policy"
  role = aws_iam_role.ecs_task.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECSExecSSM"
        Effect = "Allow"
        Action = [
          "ssmmessages:CreateControlChannel",
          "ssmmessages:CreateDataChannel",
          "ssmmessages:OpenControlChannel",
          "ssmmessages:OpenDataChannel",
        ]
        Resource = "*"
      },
      {
        Sid    = "S3Staging"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
          "s3:ListBucket",
        ]
        Resource = [
          "arn:aws:s3:::${var.project}-staging-*",
          "arn:aws:s3:::${var.project}-staging-*/*",
        ]
      },
      {
        Sid      = "SecretsManagerPrivKey"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.project}-ssh-tunnel-privkey*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/ecs/${var.project}*"
      },
      {
        Sid    = "EFSAccess"
        Effect = "Allow"
        Action = [
          "elasticfilesystem:ClientMount",
          "elasticfilesystem:ClientWrite",
        ]
        Resource = "*"
        Condition = {
          StringEquals = {
            "aws:ResourceTag/Project" = var.project
          }
        }
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# CodeBuild Service Role — builds Docker images and pushes to ECR.
# Shared across all regions.
# ---------------------------------------------------------------------------

resource "aws_iam_role" "codebuild" {
  name = "${var.project}-codebuild"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "codebuild.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "codebuild_policy" {
  name = "${var.project}-codebuild-policy"
  role = aws_iam_role.codebuild.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws/codebuild/*"
      },
      {
        Sid    = "ECR"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = "*"
      },
      {
        Sid    = "S3Source"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:GetObjectVersion",
          "s3:ListBucket",
        ]
        Resource = [
          "arn:aws:s3:::${var.project}-staging-*",
          "arn:aws:s3:::${var.project}-staging-*/*",
        ]
      },
      {
        Sid      = "SecretsManager"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.project}-dockerhub*"
      },
      {
        Sid    = "CodeBuildReports"
        Effect = "Allow"
        Action = [
          "codebuild:CreateReportGroup",
          "codebuild:CreateReport",
          "codebuild:UpdateReport",
          "codebuild:BatchPutTestCases",
          "codebuild:BatchPutCodeCoverages",
        ]
        Resource = "arn:aws:codebuild:*:${data.aws_caller_identity.current.account_id}:report-group/*"
      },
    ]
  })
}

# ---------------------------------------------------------------------------
# Orchestrator IAM user — the identity that runs NEL and orchestrates
# ECS tasks, CodeBuild builds, S3 staging, etc.  Produces AWS_ACCESS_KEY_ID
# and AWS_SECRET_ACCESS_KEY for use in CI or on a dev machine.
# Shared across all regions.
# ---------------------------------------------------------------------------

resource "aws_iam_user" "orchestrator" {
  name = "${var.project}-orchestrator"
  tags = var.tags
}

resource "aws_iam_access_key" "orchestrator" {
  user = aws_iam_user.orchestrator.name
}

resource "aws_iam_user_policy_attachment" "orchestrator" {
  user       = aws_iam_user.orchestrator.name
  policy_arn = aws_iam_policy.orchestrator.arn
}

# ---------------------------------------------------------------------------
# Orchestrator policy — permissions the orchestrator user needs across all
# regions.  Uses project-scoped wildcards so adding/removing regions doesn't
# require IAM policy changes.
# ---------------------------------------------------------------------------

resource "aws_iam_policy" "orchestrator" {
  name        = "${var.project}-orchestrator"
  description = "Permissions for the machine running NEL (orchestrator)"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid    = "ECS"
        Effect = "Allow"
        Action = [
          "ecs:RunTask",
          "ecs:StopTask",
          "ecs:DescribeTasks",
          "ecs:ExecuteCommand",
          "ecs:RegisterTaskDefinition",
          "ecs:DeregisterTaskDefinition",
          "ecs:DescribeTaskDefinition",
        ]
        Resource = "*"
      },
      {
        # AWS-default ecsTaskExecutionRole kept for nel-v1 backward compat (legacy task defs reference it).
        Sid = "PassRoles"
        Effect = "Allow"
        Action = "iam:PassRole"
        Resource = [
          aws_iam_role.ecs_execution.arn,
          aws_iam_role.ecs_task.arn,
          aws_iam_role.codebuild.arn,
          "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/service-role/ecsTaskExecutionRole",
        ]
      },
      {
        Sid    = "S3"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
        ]
        Resource = [
          "arn:aws:s3:::${var.project}-staging-*",
          "arn:aws:s3:::${var.project}-staging-*/*",
        ]
      },
      {
        Sid    = "ECR"
        Effect = "Allow"
        Action = [
          "ecr:GetAuthorizationToken",
          "ecr:BatchCheckLayerAvailability",
          "ecr:BatchGetImage",
          "ecr:DescribeImages",
          "ecr:ListImages",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
          "ecr:PutImage",
        ]
        Resource = "*"
      },
      {
        # ecs-sandbox-* prefix kept for nel-v1 backward compat (legacy CodeBuild projects).
        Sid = "CodeBuild"
        Effect = "Allow"
        Action = [
          "codebuild:CreateProject",
          "codebuild:StartBuild",
          "codebuild:BatchGetBuilds",
        ]
        Resource = [
          "arn:aws:codebuild:*:${data.aws_caller_identity.current.account_id}:project/${var.project}-*",
          "arn:aws:codebuild:*:${data.aws_caller_identity.current.account_id}:project/ecs-sandbox-*",
        ]
      },
      {
        Sid      = "SecretsManager"
        Effect   = "Allow"
        Action   = ["secretsmanager:GetSecretValue"]
        Resource = "arn:aws:secretsmanager:*:${data.aws_caller_identity.current.account_id}:secret:${var.project}-*"
      },
      {
        Sid      = "STS"
        Effect   = "Allow"
        Action   = ["sts:GetCallerIdentity"]
        Resource = "*"
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/ecs/${var.project}*"
      },
      {
        Sid      = "SSMGetParameter"
        Effect   = "Allow"
        Action   = ["ssm:GetParameter"]
        Resource = "arn:aws:ssm:*:${data.aws_caller_identity.current.account_id}:parameter/${var.project}/ecs-sandbox/*"
      },
      {
        Sid      = "EC2DescribeENI"
        Effect   = "Allow"
        Action   = ["ec2:DescribeNetworkInterfaces"]
        Resource = "*"
      },
    ]
  })

  tags = var.tags
}
