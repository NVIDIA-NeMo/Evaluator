# ---------------------------------------------------------------------------
# ECS cluster with ECS Exec (execute-command) enabled
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = "${var.project}-cluster"

  configuration {
    execute_command_configuration {
      logging = "DEFAULT"
    }
  }

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = var.tags
}

# ---------------------------------------------------------------------------
# Base ECS task definition — the orchestrator clones & overrides this per-task
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "base" {
  family                   = "${var.project}-base"
  requires_compatibilities = ["FARGATE"]
  network_mode             = "awsvpc"
  cpu                      = var.ecs_task_cpu
  memory                   = var.ecs_task_memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name      = "main"
      image     = "alpine:latest" # placeholder — overridden per-task by the orchestrator
      essential = true
      command   = ["sleep", "3600"]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.ecs.name
          "awslogs-region"        = var.region
          "awslogs-stream-prefix" = var.project
          "awslogs-create-group"  = "true"
        }
      }
    }
  ])

  tags = var.tags
}
