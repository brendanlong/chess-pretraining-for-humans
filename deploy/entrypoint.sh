#!/bin/sh
# Start the server, under Litestream when a replica is configured.
set -eu

DB="${TRAINER_DB:-/data/items.db}"

mkdir -p "$(dirname "$DB")"

# Everything past here runs as `trainer` rather than root, so a code-execution
# bug in the server doesn't also get to rewrite the image and the replica. The
# drop can't be a `USER` line in the image: only root can chown the volume, and
# Fly mounts that long after the build. It wraps Litestream as well as uvicorn —
# Litestream is the supervisor, and a child can't hold fewer privileges than the
# process that spawns it.
RUN_AS=""
if [ "$(id -u)" -eq 0 ]; then
    chown -R trainer:trainer "$(dirname "$DB")"
    RUN_AS="setpriv --reuid=trainer --regid=trainer --clear-groups --"
fi

if [ -z "${LITESTREAM_BUCKET:-}" ]; then
    # A local `podman run`, or a deployment whose secrets were never set. The
    # second is worth shouting about: it looks identical to a working one until
    # the volume is gone.
    echo "entrypoint: LITESTREAM_BUCKET is unset — NO off-machine backup" >&2
    exec $RUN_AS "$@"
fi

# -if-db-not-exists: the volume's copy is the live one and outranks the replica.
# -if-replica-exists: on the very first boot there is nothing to restore from.
$RUN_AS litestream restore -if-db-not-exists -if-replica-exists "$DB"

# `-exec` makes Litestream the supervisor: it forwards signals to the server,
# waits for it to exit, and takes a final sync on the way out (which is why
# fly.toml raises kill_timeout past Fly's 5s default). If the server dies,
# Litestream exits too, and the platform restarts the machine.
#
# `-exec` takes one string, which Litestream re-splits itself (no shell), so
# joining the arguments here round-trips only while none of them contains a
# space or a quote. That holds for the image's CMD; it's the reason the
# forwarded-IP setting is an env var rather than a `--forwarded-allow-ips '*'`
# argument, which would not have survived the trip.
exec $RUN_AS litestream replicate -exec "$*"
