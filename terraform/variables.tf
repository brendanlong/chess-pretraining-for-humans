variable "aws_region" {
  description = <<-EOT
    Region for the backup bucket. Route 53 is global. web/privacy.html says
    the app and its backups are in the United States and treats the specific
    region as ours to change — so this may move, but not out of the country
    without changing that page.
  EOT
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

# The two windows below are halves of one number the privacy policy publishes.
# A row deleted in the app is still inside every snapshot taken before the
# delete; those snapshots live for `snapshot_retention_days`, and Litestream
# then removes them, which under versioning only makes them noncurrent for a
# further `noncurrent_version_days`. The sum is how long deleted data really
# survives, and web/privacy.html tells users what that number is.

variable "snapshot_retention_days" {
  description = <<-EOT
    Point-in-time recovery window. Must equal `snapshot.retention` in
    deploy/litestream.yml, which is where Litestream actually reads it —
    this copy exists so the arithmetic below can be checked.
  EOT
  type        = number
  default     = 21
}

variable "noncurrent_version_days" {
  description = <<-EOT
    How long a backup object stays recoverable after Litestream deletes it.
    Versioning is there to survive a mistaken `rm` of the replica, and that
    mistake gets noticed in days, not months.
  EOT
  type        = number
  default     = 7

  validation {
    condition     = var.noncurrent_version_days + var.snapshot_retention_days <= var.promised_deletion_days
    error_message = <<-EOT
      Backups would outlive the promise in web/privacy.html ("deleted rows
      persist in encrypted backups for up to N days"). Shorten a window, or
      change the policy first — it is a public commitment, not a default.
    EOT
  }
}

variable "promised_deletion_days" {
  description = "The number web/privacy.html publishes. Change that page with it."
  type        = number
  default     = 30
}
