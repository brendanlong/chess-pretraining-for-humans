# These three are exactly what `fly secrets set` needs; see terraform/README.md.
output "litestream_access_key_id" {
  value = aws_iam_access_key.litestream.id
}

# The secret is in the state file either way — marking it sensitive only keeps
# it out of the console log, so treat the state bucket as a credential store.
output "litestream_secret_access_key" {
  value     = aws_iam_access_key.litestream.secret
  sensitive = true
}

output "backup_bucket" {
  value = aws_s3_bucket.backups.bucket
}

output "aws_region" {
  value = var.aws_region
}

output "hostname" {
  value = aws_route53_record.app.fqdn
}
