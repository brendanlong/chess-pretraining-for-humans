#!/usr/bin/env bash
# Refresh the deployment's item bank from a locally labeled one.
#
# Mining and labeling need Stockfish and hours of CPU, so they stay local and
# the result has to be carried over. It is carried as an items-only file and
# merged in place: the live database also holds `responses`, and replacing the
# file would destroy the experimental record. See trainer/push_items.py.
#
#   ./deploy/push-items.sh [local-db]
set -euo pipefail

REMOTE_DB=${REMOTE_DB:-/data/items.db}
REMOTE_INCOMING=${REMOTE_INCOMING:-/data/incoming-items.db}
# Absolute: an ssh session isn't the container's entrypoint and needn't have
# inherited its PATH.
REMOTE_PYTHON=${REMOTE_PYTHON:-/app/.venv/bin/python}

# Resolve the argument against the caller's directory before moving to the
# repo root, which `uv run` needs.
[ $# -gt 0 ] && LOCAL_DB=$(realpath -- "$1")
cd "$(dirname "$0")/.."
LOCAL_DB=${LOCAL_DB:-data/items.db}
[ -f "$LOCAL_DB" ] || { echo "no such database: $LOCAL_DB" >&2; exit 1; }

# -u: a name, not a file. The export refuses to write over one that exists.
export_db=$(mktemp -u -t items-export-XXXXXX.db)
trap 'rm -f "$export_db"' EXIT

uv run python -m trainer.push_items --db "$LOCAL_DB" export --out "$export_db"
# Armed before the upload, not after: a put that dies partway still leaves the
# remote file, and flyctl refuses to overwrite one, so every later run would
# fail until someone deleted it by hand. Both removals are idempotent.
trap 'rm -f "$export_db"; fly ssh console -C "rm -f $REMOTE_INCOMING" </dev/null || true' EXIT
fly ssh sftp put "$export_db" "$REMOTE_INCOMING" </dev/null

# --dry-run first: the counts are the only chance to notice you exported the
# wrong file before it's in the live bank.
#
# </dev/null on every ssh: the remote command inherits our stdin and drains it,
# so without this the dry-run swallows the answer to the prompt below and the
# read that follows sees EOF. Piping `y` in then dies at the read under `set
# -e` without printing anything, leaving the dry-run counts as the last thing
# on screen — so it reads as a completed push that never touched the bank.
fly ssh console -C "$REMOTE_PYTHON -m trainer.push_items --db $REMOTE_DB merge --dry-run $REMOTE_INCOMING" </dev/null
read -r -p "merge these into $REMOTE_DB? [y/N] " reply
if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
    fly ssh console -C "$REMOTE_PYTHON -m trainer.push_items --db $REMOTE_DB merge $REMOTE_INCOMING" </dev/null
fi
