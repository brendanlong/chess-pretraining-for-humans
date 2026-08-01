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
contexts at once. What the ledger buys is that a spent trial stays spent for
anonymous callers too: an authenticated replay is caught by the `responses` row
it already wrote, but each anonymous replay mints a fresh identity, so that row
is never there to find. Without it one held token is worth a fresh `users` row
and a fresh `responses` row per replay, up to the answer limiter's ceiling —
junk in the experimental record, under identities that answered one trial each.

The peek it thereby prices is barely worth the effort anyway, which is why
nothing more is spent on it: the reveal hands over the answer as soon as you
commit, so peeking buys, at the cost of a burnt trial, something answering would
have told you for free.

The token also carries how the trial was served — as a repeat, and from a share
link — so both are decided from what the server offered rather than from what
the client says or from what the bank happens to look like when the answer
arrives. Deciding the repeat from the bank at redemption time gets both
boundaries wrong: answering your last unseen item drops the count to zero and
makes that token replayable, and a bank refilled mid-trial makes a
legitimately-served repeat unanswerable. And a share is a claim about the
request that fetched the trial, which the request that answers it can't see;
taking the client's word for it would buy a caller the calibration staircase's
exemption on demand, and leave a friend's link unmarked in the research record.

Changing the payload's shape invalidates the tokens in flight, exactly as
rotating the key does: they stop verifying, the client is told 409, and it
fetches a trial it can answer.
"""

import base64
import hashlib
import hmac
import logging
import os
import secrets
import time
from typing import NamedTuple

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


class Served(NamedTuple):
    """How the server offered a trial, read back off its own token.

    `repeat` is "answered before", so the answer earns feedback and moves
    nothing — either the bank ran out, or a URL named a position this caller
    has already answered. `shared` is a URL naming the item rather than
    selection choosing it; on a first exposure the answer counts like any
    other, but it is marked in the record, and it is scored by Elo rather than
    by the calibration staircase, which only means anything on an item aimed at
    the user. The two are independent: a reopened link is both.
    """

    repeat: bool
    shared: bool


def _sign(payload: str) -> str:
    mac = hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


FIELDS = 6  # item, user, repeat, shared, nonce, expiry — then the mac


def issue(item_id: int, user_id: int | None, served: Served) -> str:
    """A token asserting that `item_id` was offered to `user_id` (0 = nobody yet,
    which is every trial served before its owner has answered anything), and how
    it was offered."""
    ttl = TOKEN_TTL_S if user_id else ANON_TOKEN_TTL_S
    # The nonce makes every issuance distinct. Without it, two anonymous callers
    # served the same item in the same second get byte-identical tokens — and a
    # spend-once ledger keyed on the token would let the first of them answer and
    # refuse the second, which is two strangers colliding rather than a replay.
    nonce = secrets.token_urlsafe(8)
    payload = (
        f"{item_id}.{user_id or 0}.{int(served.repeat)}.{int(served.shared)}"
        f".{nonce}.{int(time.time()) + ttl}"
    )
    return f"{payload}.{_sign(payload)}"


def redeem(token: str | None, item_id: int, user_id: int | None) -> Served:
    """Raise unless `token` is this server's proof that it offered `item_id` to
    `user_id`; return how it offered it.

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
        token_item, token_user, repeat, shared = (int(p) for p in parts[:4])
        expires = int(parts[5])
    except ValueError as e:  # we signed it, so this is corruption, not an attack
        raise InvalidTrial("unreadable trial token") from e
    if token_item != item_id:
        raise InvalidTrial("trial token is for a different item")
    if token_user != (user_id or 0):
        raise InvalidTrial("trial token was issued to a different session")
    if time.time() >= expires:
        raise InvalidTrial("trial token has expired")
    return Served(repeat=bool(repeat), shared=bool(shared))
