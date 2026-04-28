# ---------------------------------------------------------------------------
# Default provider — used for global resources (IAM).
# IAM is region-agnostic; we use the first configured region by convention.
# ---------------------------------------------------------------------------

provider "aws" {
  region = var.regions[0]
}

data "aws_caller_identity" "current" {}

# ---------------------------------------------------------------------------
# Aliased providers — one per deployable region.  Each regional module
# instance receives the matching alias via its providers block.
# ---------------------------------------------------------------------------

provider "aws" {
  alias  = "us_east_1"
  region = "us-east-1"
}

provider "aws" {
  alias  = "us_east_2"
  region = "us-east-2"
}

provider "aws" {
  alias  = "us_west_1"
  region = "us-west-1"
}

provider "aws" {
  alias  = "us_west_2"
  region = "us-west-2"
}

provider "aws" {
  alias  = "ca_central_1"
  region = "ca-central-1"
}

provider "aws" {
  alias  = "eu_west_1"
  region = "eu-west-1"
}

provider "aws" {
  alias  = "eu_west_2"
  region = "eu-west-2"
}

provider "aws" {
  alias  = "eu_west_3"
  region = "eu-west-3"
}

provider "aws" {
  alias  = "eu_central_1"
  region = "eu-central-1"
}

provider "aws" {
  alias  = "eu_north_1"
  region = "eu-north-1"
}

provider "aws" {
  alias  = "sa_east_1"
  region = "sa-east-1"
}
