# Deploying

One Fly machine, the SQLite file on a Fly volume, Litestream streaming that
file to S3, DNS and the bucket in Terraform. The item bank is rebuildable from
the pipeline; `responses` is not, which is why there's a backup story at all.

| File | What it is |
| --- | --- |
| `../Dockerfile` | uv build of the server and esbuild of the frontend; no Stockfish, no zstd |
| `../fly.toml` | one machine, volume at `/data`, health check on `/healthz` |
| `entrypoint.sh` | restore-if-empty, then run uvicorn under Litestream |
| `litestream.yml` | replication config (`/etc/litestream.yml` in the image) |
| `push-items.sh` | ship a locally labeled bank to the deployment |
| `../terraform/` | Route 53 record, backup bucket, Litestream's IAM user |
| `../.github/workflows/deploy.yml` | `flyctl deploy` after CI goes green on main |

## Behind a reverse proxy (Fly's, or your own)

Terminate TLS at the proxy (the session cookie is marked `Secure` whenever
the request arrives over https) and run uvicorn with `--proxy-headers` and a
trusted `--forwarded-allow-ips` — the Dockerfile and `fly.toml` already do.

That settles the scheme but not the address: with `--forwarded-allow-ips '*'`
uvicorn believes the leftmost `X-Forwarded-For` entry, and proxies append to
that header rather than replacing it, so the address is whatever the caller
put there. Set `CLIENT_IP_HEADER` to a header your proxy *overwrites*
(`fly-client-ip` on Fly, or an `X-Real-IP` you set yourself in nginx) and the
per-address limits are charged to that instead. Leave it unset when nothing
is in front. Login throttling is per account and unaffected either way.

If you expose this publicly, put per-IP request limiting in the proxy
(`limit_req` in nginx, or equivalent) rather than relying on the in-app
counter: the proxy's is shared across workers, survives a restart, and sheds
load before it reaches Python.

## One-time bootstrap

Order matters: Terraform mints the AWS key that Fly needs as a secret, and Fly
issues the certificate only once DNS points at it.

**1. AWS.** The Terraform state bucket has to exist before `init` can use it —
see [terraform/README.md](../terraform/README.md) for that one-liner and the
rest. Then:

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # already the live values
terraform init
terraform apply
```

**2. Fly.** `-s` is in GB, and it's the volume, not the machine, that has to be
sized: the bank, its WAL, and — during a refresh — a second full copy of the
bank uploaded next to it. Budget twice the bank and then some; a volume that
fills mid-merge gives the live database `SQLITE_FULL`.

```bash
fly apps create chess-pretraining --org personal
fly volumes create chess_pretraining_data -a chess-pretraining -r iad -s 1
fly secrets set -a chess-pretraining \
  AWS_ACCESS_KEY_ID="$(cd terraform && terraform output -raw litestream_access_key_id)" \
  AWS_SECRET_ACCESS_KEY="$(cd terraform && terraform output -raw litestream_secret_access_key)" \
  AWS_REGION="$(cd terraform && terraform output -raw aws_region)" \
  LITESTREAM_BUCKET="$(cd terraform && terraform output -raw backup_bucket)" \
  TRIAL_TOKEN_SECRET="$(openssl rand -hex 32)"
fly deploy --ha=false
```

`TRIAL_TOKEN_SECRET` signs the trial tokens that let `/api/answer` tell a trial
the server offered from an item id somebody typed. It isn't a user credential
and nothing is stored under it, so rotating it is free apart from refusing the
trials in flight at that moment — clients fetch a fresh one. Unset, the server
generates an ephemeral key and logs that it did: fine on a laptop, and on a
machine that restarts it means every open tab gets one refused answer.

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

The default expiry is 20 years, so `-x 8760h` is worth passing — but a token
that expires is a deploy that starts failing in CI rather than a site that goes
down, and nothing warns you beforehand. `fly tokens list -a chess-pretraining`
prints the expiry; the live one runs to 2027-07-30.

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
from a bad write means restoring alongside and then swapping — **with the
machine stopped**. Replacing the file under a running Litestream is the one
mistake with no feedback: it keeps reporting healthy syncs against a database
it no longer understands (verified — copying over the file produced no error
and no new transaction), so the replica quietly stops matching reality.

Do the swap from inside the machine and let the restart reuse the boot-time
restore — `fly ssh` talks to an agent *in* the machine, so anything that needs
a shell has to happen before you stop it.

```bash
fly ssh console -a chess-pretraining
  # Restore beside the live file and check it's the state you wanted:
  /usr/local/bin/litestream restore -config /etc/litestream.yml \
    -o /data/restored.db -timestamp 2026-07-28T12:00:00Z /data/items.db
  # (no sqlite3 CLI in the image — it's a server, not a toolbox)
  /app/.venv/bin/python -c "import sqlite3; print(sqlite3.connect(
    '/data/restored.db').execute('SELECT COUNT(*) FROM responses').fetchone())"
  # Then take the live file out of the way. The metadata directory describes
  # the file you're removing, so it goes too.
  mv /data/items.db /data/items.db.bad && rm -rf /data/.items.db-litestream
  exit
fly machine restart <id>   # boots into the entrypoint's restore
```

Restarting rather than moving `restored.db` into place is deliberate: it goes
through the same path a lost volume does, which is the path that gets
exercised. Keep `items.db.bad` until you're satisfied, then delete it — it's
on the volume and Litestream is not replicating it.

### How far back you can go, and why it stops there

Litestream's own default retention is 24h, which would make a Friday mistake
unrecoverable by Monday. But the window has a ceiling as well as a floor:
`web/privacy.html` tells users that a deleted account survives in backups for
at most 30 days, and a deleted row sits in every snapshot taken before the
delete. So the recovery window is 21 days (`snapshot.retention` in
`litestream.yml`) plus 7 for the bucket's noncurrent versions — 28, under the
promise. `terraform/variables.tf` checks that sum at plan time; raising either
half fails the plan and points here. The published number is the thing to
change first, if it should change.

## Things that will bite

- **One machine, always.** `fly deploy --ha=false`, and don't `fly scale count`
  past 1. Two machines means two volumes and two Litestreams on one S3 prefix.
  `min_machines_running = 1` in `fly.toml` doesn't create one; it's a floor the
  proxy won't stop below, so the machine stays up instead of idling out while
  someone thinks over a position. It counts only machines in `primary_region`,
  which is fine while that's where the volume forces the machine to live.
- **Migrations run on connect, and some of them drop columns — which makes
  those releases one-way.** Rolling back past one means restoring the database
  too, not just `flyctl releases rollback`: the older server's `INSERT` names
  columns the newer schema no longer has, so it boots happily and then 500s on
  every answer. Anything a dropped column held is gone from the live file the
  first time the new server opens it, so copy it out of a Litestream restore
  beforehand if you want it.
- **A database older than the current schema will not open.** Items carry the
  measurement their difficulty is derived from, and `db.connect` refuses a bank
  without it rather than serving items at whatever an older curve left behind:

      RuntimeError: /data/items.db has items with no shallow_gap …

  A restore from far enough back therefore needs a bank pushed over it (above)
  before the machine will serve. That is the deliberate trade for the app
  carrying no code to tolerate a half-measured bank.
- **`VACUUM` breaks replication.** It rewrites every page and invalidates
  Litestream's tracking. `VACUUM INTO` a new file and treat it as a new
  database (stop Litestream, clear `/data/.items.db-litestream`, re-snapshot).
- **Litestream adds `_litestream_seq` and `_litestream_lock` tables** to the
  database. They're expected; nothing in the app enumerates tables, but a
  schema diff against a local copy will show them.
- **Replication is asynchronous** (1s). Losing the host loses about a second of
  answers. A clean stop syncs, which is what `kill_timeout = 30` protects.
- **The page counter's settings live in its dashboard, not here.** Leave
  "collect individual pageviews" off: the privacy policy says what's stored is
  counts, and that switch makes it a row per visit. Its script is pinned to a
  version and an `integrity` hash, so a newer one means changing both in all
  three `web/*.html` heads — a mismatched hash fails silently.
- **Anything on a public hostname counts into the live dashboard.** The
  counter skips only localhost and private ranges, so a staging deploy or a
  laptop reached over a VPN name reports real hits.
- **Local testing** doesn't need Fly. `podman build -t chess . && podman run
  -p 8000:8080 -v ./data:/data:Z chess` runs the same image without a replica.
