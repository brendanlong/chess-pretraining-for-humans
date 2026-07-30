"""Trial tokens: the server's own proof that it offered this item to this caller.

`/api/answer` returns the answer key, so it has to be reachable only by
answering a trial that was really served. Recording that server-side would mean
a `users` row exists before the first trial — and minting a row on arrival
makes arriving a write, a write needs a limit, and a limit on arriving is a gate
in front of the first trial, which SPEC forbids. A signed token needs no row:
the pending trial becomes state the client holds and cannot forge.

The token names the holder as well as the item. That part is load-bearing: a
token issued to nobody in particular could otherwise be fetched by a throwaway
cookieless client and spent by the signed-in one, which is the pre-commit peek it
exists to prevent.

A token issued before its holder has any identity can't be bound to a session
that doesn't exist yet, so the server remembers it has spent one instead — see
`server.anonymous_trial_use`. Note who that defends against: a token is only ever
handed to the one client that asked for it, over TLS, in a `no-store` body, never
in a URL or a log, so nobody else has it. The actor is the holder using two
contexts at once, and the thing worth stopping is not the peek — replaying is how
one client mints a fresh row per replay, and since each row is seeing the item for
the first time, each replay counts as a first exposure and moves the item's shared
counters. That is a targeted skew of one item's difficulty, which is worse than
the diffuse noise `next`→`answer` can already make.

The peek itself barely repays the effort, which is why nothing more is spent on
it: the reveal hands over the answer as soon as you commit, so peeking buys, at
the cost of a burnt trial, something answering would have told you for free.

The token also carries whether the trial was served *as a repeat*, so the "you
already answered this" rule is decided from what the server offered rather than
from what the bank happens to look like when the answer arrives. Deciding it
from the bank at redemption time gets both boundaries wrong: answering your
last unseen item drops the count to zero and makes that token replayable, and a
bank refilled mid-trial makes a legitimately-served repeat unanswerable.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time

log = logging.getLogger(__name__)

# Generous, because a tab someone walked away from should still be answerable
# when they come back; bounded, because a token shouldn't be a durable artifact.
# Expired means "fetch a new trial", which is what the client does with a 409.
TOKEN_TTL_S = 12 * 3600
# A token issued before its holder has any identity gets much less, because it is
# the weaker kind: anonymous tokens are interchangeable, so the only thing
# stopping a replay is the server remembering it. A short life keeps that set
# small, and nobody's *first* answer is twelve hours after their first trial.
ANON_TOKEN_TTL_S = 900

KEY_ENV_NAME = "TRIAL_TOKEN_SECRET"


def _load_secret() -> bytes:
    configured = os.environ.get(KEY_ENV_NAME, "")
    if configured:
        return configured.encode()
    # A laptop or a test run. An ephemeral key behaves exactly like a configured
    # one until the process restarts, which rejects the trials in flight — fine
    # locally, and the log line is here so that it is never quietly the case in
    # production, where `fly secrets set TRIAL_TOKEN_SECRET=…` supplies one.
    #
    # The variable's name is spelled out rather than interpolated from
    # KEY_ENV_NAME on purpose: passing a value named "…SECRET…" into a logger is
    # the exact shape of a real credential leak, so both CodeQL and a human
    # skimming this have to stop and check that it's only ever the name. Don't
    # helpfully refactor it back.
    log.warning(
        "TRIAL_TOKEN_SECRET is unset — using an ephemeral key. Trials in flight "
        "will be refused across a restart."
    )
    return secrets.token_bytes(32)


# Read once per process. One uvicorn worker today (see the Dockerfile CMD), and
# a configured key would keep several in agreement anyway — but an *ephemeral*
# key plus `--workers N` would hand out tokens each worker alone can verify, so
# the two changes have to arrive together.
SECRET = _load_secret()


class InvalidTrial(Exception):
    """The token is missing, malformed, forged, expired, or someone else's."""


def _sign(payload: str) -> str:
    mac = hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


FIELDS = 5  # item, user, served-as-repeat, nonce, expiry — then the mac


def issue(item_id: int, user_id: int | None, served_as_repeat: bool) -> str:
    """A token asserting that `item_id` was offered to `user_id` (0 = nobody yet,
    which is every trial served before its owner has answered anything), and
    whether it was offered as a repeat."""
    ttl = TOKEN_TTL_S if user_id else ANON_TOKEN_TTL_S
    # The nonce makes every issuance distinct. Without it, two anonymous callers
    # served the same item in the same second get byte-identical tokens — and a
    # spend-once ledger keyed on the token would let the first of them answer and
    # refuse the second, which is two strangers colliding rather than a replay.
    nonce = secrets.token_urlsafe(8)
    payload = f"{item_id}.{user_id or 0}.{int(served_as_repeat)}.{nonce}.{int(time.time()) + ttl}"
    return f"{payload}.{_sign(payload)}"


def redeem(token: str | None, item_id: int, user_id: int | None) -> bool:
    """Raise unless `token` is this server's proof that it offered `item_id` to
    `user_id`; return whether it offered it as a repeat.

    Every field is signed, so all of this is the server reading back its own
    claim. Nothing is consumed here — spending an anonymous token is the caller's
    job, because only it knows whether the answer went on to be recorded.
    """
    parts = (token or "").split(".")
    if len(parts) != FIELDS + 1:
        raise InvalidTrial("malformed trial token")
    payload, mac = ".".join(parts[:FIELDS]), parts[FIELDS]
    # compare_digest before parsing anything: the fields are only meaningful
    # once we know we wrote them.
    if not hmac.compare_digest(mac, _sign(payload)):
        raise InvalidTrial("trial token does not verify")
    try:  # the nonce is opaque, and only ever compared as part of the payload
        token_item, token_user, served_as_repeat = (int(p) for p in parts[:3])
        expires = int(parts[4])
    except ValueError as e:  # we signed it, so this is corruption, not an attack
        raise InvalidTrial("unreadable trial token") from e
    if token_item != item_id:
        raise InvalidTrial("trial token is for a different item")
    if token_user != (user_id or 0):
        raise InvalidTrial("trial token was issued to a different session")
    if time.time() >= expires:
        raise InvalidTrial("trial token has expired")
    return bool(served_as_repeat)
