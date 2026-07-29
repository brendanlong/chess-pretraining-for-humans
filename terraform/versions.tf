terraform {
  required_version = ">= 1.10"

  # Partial config: the bucket and key come from backend.hcl, which isn't
  # committed because the state holds the Litestream secret key in clear text.
  #   terraform init -backend-config=backend.hcl
  #
  # `use_lockfile` is S3's own conditional-write lock (Terraform >= 1.10), so
  # there is no DynamoDB table to create and pay for.
  backend "s3" {
    encrypt      = true
    use_lockfile = true
  }

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}
