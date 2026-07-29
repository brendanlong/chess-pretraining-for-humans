variable "aws_region" {
  description = "Region for the backup bucket. Route 53 is global."
  type        = string
  default     = "us-east-1"
}

variable "zone_name" {
  description = "Existing Route 53 hosted zone the site's name lives under. Not managed here."
  type        = string
  default     = "brendanlong.com"
}

variable "hostname" {
  description = "Public name for the app."
  type        = string
  default     = "chess-pretraining.brendanlong.com"
}

variable "fly_hostname" {
  description = <<-EOT
    The app's Fly-assigned name, e.g. "chess-pretraining.fly.dev". A CNAME to
    it is enough: it resolves to whatever addresses Fly currently has for the
    app, so a change on their side needs no change here, and it lets Fly issue
    the certificate over HTTP validation without a DNS challenge.
  EOT
  type        = string
}

variable "acme_challenge_target" {
  description = <<-EOT
    Only if `fly certs check` asks for a DNS challenge (it does for a wildcard,
    and can if HTTP validation is blocked): the CNAME target it prints for
    `_acme-challenge.<hostname>`. Empty means no such record.
  EOT
  type        = string
  default     = ""
}

variable "backup_bucket" {
  description = "Bucket holding the Litestream replica of the SQLite database."
  type        = string
}

variable "backup_retention_days" {
  description = <<-EOT
    How long a deleted or superseded backup object stays recoverable. This is
    the window for noticing a mistake, so it wants to be longer than a weekend.
  EOT
  type        = number
  default     = 90
}
