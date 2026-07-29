# A Fly volume is one disk on one host. `responses` is the experimental record
# and can't be regenerated from anything, so it is replicated continuously to
# S3 (see deploy/litestream.yml) rather than trusted to that disk.

resource "aws_s3_bucket" "backups" {
  bucket = var.backup_bucket

  lifecycle {
    prevent_destroy = true
  }
}

# Litestream's own generations survive an accidental `rm` of the replica, but
# only if the delete is recoverable — hence versioning, not just the bucket.
resource "aws_s3_bucket_versioning" "backups" {
  bucket = aws_s3_bucket.backups.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backups" {
  bucket                  = aws_s3_bucket.backups.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id

  # Litestream writes small WAL segments continuously and retires old
  # generations itself; what needs a rule is the debris that leaves behind.
  rule {
    id     = "expire-old-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = var.backup_retention_days
    }
    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.backups]
}

# Fly machines have no way to assume an AWS role, so replication authenticates
# with a static key. Keep it to one bucket and to the actions Litestream uses.
resource "aws_iam_user" "litestream" {
  name = "litestream-${replace(var.backup_bucket, ".", "-")}"
}

data "aws_iam_policy_document" "litestream" {
  statement {
    sid       = "InspectTheBucket"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [aws_s3_bucket.backups.arn]
  }
  statement {
    sid = "ReadWriteTheReplica"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      # Litestream retires superseded snapshots and WAL segments as it goes,
      # and releases its single-writer lease by deleting a lock object.
      "s3:DeleteObject",
      # Not in Litestream's published minimum policy, but its uploader aborts
      # failed multipart uploads; without this those fail and leave orphaned
      # parts behind (which the lifecycle rule above then has to sweep).
      "s3:AbortMultipartUpload",
    ]
    resources = ["${aws_s3_bucket.backups.arn}/*"]
  }
}

resource "aws_iam_user_policy" "litestream" {
  name   = "litestream-replica"
  user   = aws_iam_user.litestream.name
  policy = data.aws_iam_policy_document.litestream.json
}

resource "aws_iam_access_key" "litestream" {
  user = aws_iam_user.litestream.name
}
