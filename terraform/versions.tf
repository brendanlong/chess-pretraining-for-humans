terraform {
  required_version = ">= 1.10"

  # The state holds the Litestream secret key in clear text, but that is an
  # argument for locking the bucket down, not for hiding its name — so the
  # backend is fully declared here and `terraform init` needs no extra flag.
  # Guarding the contents is the bucket's own policy; see README.md.
  #
  # `use_lockfile` is S3's own conditional-write lock (Terraform >= 1.10), so
  # there is no DynamoDB table to create and pay for.
  backend "s3" {
    bucket       = "chess-pretraining-state"
    key          = "chess-pretraining/terraform.tfstate"
    region       = "us-east-1"
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
