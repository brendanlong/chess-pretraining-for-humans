# Terraform — the AWS half

Fly hosts the app; AWS holds the name and the backups. Two things live here:

- a Route 53 record in an **existing** hosted zone (the zone is looked up, not
  managed — it's shared with everything else on the domain), and
- the S3 bucket Litestream replicates into, plus the IAM user it authenticates
  as. Fly machines can't assume a role, so that's a long-lived access key,
  scoped to this one bucket.

The bucket's retention is not a free choice: `web/privacy.html` publishes how
long a deleted account persists in backups, and a deleted row is inside every
snapshot older than the delete. `snapshot_retention_days` (which must match
`deploy/litestream.yml`) plus `noncurrent_version_days` is that real lifetime,
and a validation fails the plan if the two exceed what the policy promises.

Fly itself is not in here: its Terraform provider is archived, and `fly.toml`
plus `flyctl` is the supported path. See [../deploy/README.md](../deploy/README.md).

## Running it

The state bucket has to exist before `init` can lock against it. Create it once
by hand — versioning is the only undo you get for a bad state write:

```bash
B=your-terraform-state-bucket; R=us-east-1
aws s3api create-bucket --bucket "$B" --region "$R"
aws s3api put-bucket-versioning --bucket "$B" --versioning-configuration Status=Enabled
aws s3api put-public-access-block --bucket "$B" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$B" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"}}]}'
```

(Outside `us-east-1`, `create-bucket` also needs
`--create-bucket-configuration LocationConstraint="$R"`.)

Then:

```bash
cp backend.hcl.example backend.hcl
cp terraform.tfvars.example terraform.tfvars
$EDITOR backend.hcl terraform.tfvars
terraform init -backend-config=backend.hcl
terraform plan
terraform apply
```

Locking is S3's own conditional write (`use_lockfile`, Terraform ≥ 1.10) — a
`.tflock` object, no DynamoDB table.

## Handing the key to Fly

```bash
fly secrets set -a chess-pretraining \
  AWS_ACCESS_KEY_ID="$(terraform output -raw litestream_access_key_id)" \
  AWS_SECRET_ACCESS_KEY="$(terraform output -raw litestream_secret_access_key)" \
  AWS_REGION="$(terraform output -raw aws_region)" \
  LITESTREAM_BUCKET="$(terraform output -raw backup_bucket)"
```

**The secret key is stored in the Terraform state in clear text.** There is no
write-only variant of a generated attribute, so protecting the state bucket is
the whole defence — keep it private, versioned, and out of any shared account.
Rotating is `terraform apply -replace=aws_iam_access_key.litestream`, followed
by setting the secret on Fly again. That apply destroys the old key as it
creates the new one, so replication gets 403s until the `fly secrets set`
lands and the machine restarts — it catches up afterwards, but don't rotate
and then walk away.

## Why a CNAME rather than A + AAAA

The record points at `<app>.fly.dev` instead of at Fly's addresses. It's a
subdomain, so a CNAME is legal, and it means Fly can change the anycast IPs
(the shared IPv4 in particular) without this repo going stale. An apex would
have to use A + AAAA from `fly ips list`, and would then want the
`_acme-challenge` record — `acme_challenge_target` covers that case.

## Not managed here

Domain registration (neither Terraform nor CloudFormation can register a
domain from scratch), the hosted zone itself, and every other record in it.
Terraform only touches what it declares, so the rest of the zone is safe.
