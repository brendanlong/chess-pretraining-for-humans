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

LOCAL_DB=${1:-data/items.db}
REMOTE_DB=${REMOTE_DB:-/data/items.db}
REMOTE_INCOMING=${REMOTE_INCOMING:-/data/incoming-items.db}
# Absolute: an ssh session isn't the container's entrypoint and needn't have
# inherited its PATH.
REMOTE_PYTHON=${REMOTE_PYTHON:-/app/.venv/bin/python}

cd "$(dirname "$0")/.."
[ -f "$LOCAL_DB" ] || { echo "no such database: $LOCAL_DB" >&2; exit 1; }

export_db=$(mktemp -u -t items-export-XXXXXX.db)
trap 'rm -f "$export_db"' EXIT

uv run python -m trainer.push_items --db "$LOCAL_DB" export --out "$export_db"
fly ssh sftp put "$export_db" "$REMOTE_INCOMING"

# --dry-run first: the counts are the only chance to notice you exported the
# wrong file before it's in the live bank.
fly ssh console -C "$REMOTE_PYTHON -m trainer.push_items --db $REMOTE_DB merge --dry-run $REMOTE_INCOMING"
read -r -p "merge these into $REMOTE_DB? [y/N] " reply
if [ "$reply" = "y" ] || [ "$reply" = "Y" ]; then
    fly ssh console -C "$REMOTE_PYTHON -m trainer.push_items --db $REMOTE_DB merge $REMOTE_INCOMING"
fi
fly ssh console -C "rm -f $REMOTE_INCOMING"
