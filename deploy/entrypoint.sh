#!/bin/sh
# Start the server, under Litestream when a replica is configured.
set -eu

DB="${TRAINER_DB:-/data/items.db}"

if [ -z "${LITESTREAM_BUCKET:-}" ]; then
    # A local `podman run`, or a deployment whose secrets were never set. The
    # second is worth shouting about: it looks identical to a working one until
    # the volume is gone.
    echo "entrypoint: LITESTREAM_BUCKET is unset — NO off-machine backup" >&2
    exec "$@"
fi

mkdir -p "$(dirname "$DB")"
# -if-db-not-exists: the volume's copy is the live one and outranks the replica.
# -if-replica-exists: on the very first boot there is nothing to restore from.
litestream restore -if-db-not-exists -if-replica-exists "$DB"

# `-exec` makes Litestream the supervisor: it forwards signals to the server,
# waits for it to exit, and takes a final sync on the way out (which is why
# fly.toml raises kill_timeout past Fly's 5s default). If the server dies,
# Litestream exits too, and the platform restarts the machine.
exec litestream replicate -exec "$*"
