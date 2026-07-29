# Deploying

One Fly machine, the SQLite file on a Fly volume, Litestream streaming that
file to S3, DNS and the bucket in Terraform. The item bank is rebuildable from
the pipeline; `responses` is not, which is why there's a backup story at all.

| File | What it is |
| --- | --- |
| `../Dockerfile` | uv build of the server; no Stockfish, no zstd |
| `../fly.toml` | one machine, volume at `/data`, health check on `/healthz` |
| `entrypoint.sh` | restore-if-empty, then run uvicorn under Litestream |
| `litestream.yml` | replication config (`/etc/litestream.yml` in the image) |
| `push-items.sh` | ship a locally labeled bank to the deployment |
| `../terraform/` | Route 53 record, backup bucket, Litestream's IAM user |
| `../.github/workflows/deploy.yml` | `flyctl deploy` after CI goes green on main |

## One-time bootstrap

Order matters: Terraform mints the AWS key that Fly needs as a secret, and Fly
issues the certificate only once DNS points at it.

**1. AWS.** The Terraform state bucket has to exist before `init` can use it —
see [terraform/README.md](../terraform/README.md) for that one-liner and the
rest. Then:

```bash
cd terraform
cp backend.hcl.example backend.hcl && cp terraform.tfvars.example terraform.tfvars
$EDITOR backend.hcl terraform.tfvars     # bucket names, the Fly app's hostname
terraform init -backend-config=backend.hcl
terraform apply
```

**2. Fly.** `-s` is in GB, and it's the volume, not the machine, that has to be
big enough for the bank plus its WAL.

```bash
fly apps create chess-pretraining --org personal
fly volumes create chess_pretraining_data -a chess-pretraining -r iad -s 1
fly secrets set -a chess-pretraining \
  AWS_ACCESS_KEY_ID="$(cd terraform && terraform output -raw litestream_access_key_id)" \
  AWS_SECRET_ACCESS_KEY="$(cd terraform && terraform output -raw litestream_secret_access_key)" \
  AWS_REGION="$(cd terraform && terraform output -raw aws_region)" \
  LITESTREAM_BUCKET="$(cd terraform && terraform output -raw backup_bucket)"
fly deploy --ha=false
```

Without `LITESTREAM_BUCKET` the container still starts — it just logs that it
has no off-machine backup and serves anyway. That's deliberate (a broken
replica shouldn't take the site down) and it means the log line is the only
thing standing between you and an unreplicated deployment. Check for it.

**3. TLS.** The Terraform CNAME points the name at `<app>.fly.dev`, which is
enough for Fly to validate over HTTP:

```bash
fly certs add chess-pretraining.brendanlong.com
fly certs check chess-pretraining.brendanlong.com
```

If `check` asks for a DNS challenge instead, put the target it prints into
`acme_challenge_target` in `terraform.tfvars` and apply again.

**4. Continuous deploys.** `fly tokens create deploy` prints an app-scoped
token; add it as the `FLY_API_TOKEN` repository secret. Pushes to main then
deploy once CI passes.

**5. Fill the bank.** A fresh deployment has no items and `/api/next` answers
503 until it does — see below.

## Refreshing the item bank

Mining and labeling stay local. The live database holds the responses in the
same file, so the bank is merged in, never copied over:

```bash
./deploy/push-items.sh data/items.db
```

It exports an items-only database, uploads it, shows you what a merge would
add, asks, then merges. Positions already in the bank are skipped rather than
relabelled: an item whose best move changed under the answers already given to
it would make those responses uninterpretable.

## Restoring

Litestream restores on boot only when the database is missing, so recovering
from a bad write means deliberately moving the live file aside first — on the
machine, with the server stopped, since a whole-file replace under a running
Litestream desyncs it.

```bash
fly ssh console -a chess-pretraining
/usr/local/bin/litestream restore -o /data/restored.db \
  -timestamp 2026-07-28T12:00:00Z /data/items.db
```

Snapshot retention is 720h (`litestream.yml`); Litestream's own default is 24h,
which would have meant a Friday mistake was unrecoverable by Monday.

## Things that will bite

- **One machine, always.** `fly deploy --ha=false`, and don't `fly scale count`
  past 1. Two machines means two volumes and two Litestreams on one S3 prefix.
- **`VACUUM` breaks replication.** It rewrites every page and invalidates
  Litestream's tracking. `VACUUM INTO` a new file and treat it as a new
  database (stop Litestream, clear `/data/.items.db-litestream`, re-snapshot).
- **Litestream adds `_litestream_seq` and `_litestream_lock` tables** to the
  database. They're expected; nothing in the app enumerates tables, but a
  schema diff against a local copy will show them.
- **Replication is asynchronous** (1s). Losing the host loses about a second of
  answers. A clean stop syncs, which is what `kill_timeout = 30` protects.
- **Local testing** doesn't need Fly. `podman build -t chess . && podman run
  -p 8000:8080 -v ./data:/data:Z chess` runs the same image without a replica.
