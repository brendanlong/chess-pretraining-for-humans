# The zone is shared with everything else on the domain, so it is looked up,
# never declared: Terraform only owns the records it names here.
data "aws_route53_zone" "root" {
  name         = "${var.zone_name}."
  private_zone = false
}

resource "aws_route53_record" "app" {
  zone_id = data.aws_route53_zone.root.zone_id
  name    = var.hostname
  type    = "CNAME"
  ttl     = 300
  records = [var.fly_hostname]
}

# `fly certs add` prints this when it wants DNS validation rather than HTTP.
resource "aws_route53_record" "acme_challenge" {
  count = var.acme_challenge_target == "" ? 0 : 1

  zone_id = data.aws_route53_zone.root.zone_id
  name    = "_acme-challenge.${var.hostname}"
  type    = "CNAME"
  ttl     = 300
  records = [var.acme_challenge_target]
}
