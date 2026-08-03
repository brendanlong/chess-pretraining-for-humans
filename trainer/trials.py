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

Expiry alone is recoverable, and `reissue` is what recovers it: the clock is
the one thing a stale token fails on that says nothing about who is holding it
or what they were offered. So an expired token is re-signed rather than
refused, the client answers the trial it was actually looking at, and nothing
about the position had to be handed out a second time — `reissue` returns a
token and only a token. The rest are refusals on purpose. A token that names
another session is another person's pick, and replaying it would file it under
this one; a token we can't verify at all — a rotated key, a changed payload
shape — tells us nothing to re-sign, and guessing would mean re-deriving how
the trial was served from a caller who could then choose it. What survives a
key change is therefore the same as before: the client is told to fetch a
trial it can answer, and this one's answer is lost.

Two things keep re-signing from re-arming a spent anonymous trial, and both are
load-bearing. Re-signing keeps the nonce, and the ledger is keyed on that
rather than on the token carrying it, so every token naming one trial shares
one slot. And the trial has a life of its own (`TRIAL_LIFE_S`) that no
re-signing extends, so the ledger can be sized to outlive every token that can
name it: a spend it has forgotten is a spend nothing can still be redeemed
against. Without that second half, re-signing would be exactly the replay the
ledger exists to stop — one held token, a fresh guest and a fresh first
exposure against the same item for as long as the caller cared to keep asking.
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

# How long a *trial* can be answered for, measured from the moment it was
# offered and carried in the token so that re-signing one cannot extend it. A
# day, because the tab that comes back is the one that was slept on. This is the
# bound the spend-once ledger is sized against, so raising it costs memory
# there; it is not a bound on anything else, since a signed-in caller's replay
# is caught by the row they already wrote.
TRIAL_LIFE_S = 24 * 3600
# How long one token stands for that trial before it has to be re-signed.
# Generous, because a tab someone walked away from should still be answerable
# when they come back; bounded, because it is the window in which a token that
# somehow got loose is worth something.
TOKEN_TTL_S = 12 * 3600
# A token issued before its holder has any identity gets much less, because it
# is the weaker kind: anonymous tokens are interchangeable, so the only thing
# stopping a replay is the server remembering it. Short is affordable because
# expiry costs a round trip and not an answer — the tab that comes back after
# lunch re-signs its token and answers the position it was looking at — and it
# keeps the window in which a loose one is spendable down to minutes.
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


class TrialExpired(InvalidTrial):
    """Ours, this caller's, this item's — and only the clock is against it.

    Split out because it is the one failure the holder can be handed a fix for:
    everything the new token would say is readable off the old one. The
    distinction has to reach the client, which is why `/api/answer` answers this
    with its own status.
    """


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


FIELDS = 7  # item, user, repeat, shared, nonce, offered, expiry — then the mac
NONCE_FIELD = 4


class Trial(NamedTuple):
    """A verified token's contents, minus the parts the caller supplied."""

    served: Served
    # Which offer this is, rather than which token names it: re-signing carries
    # it over, so it identifies the trial across however many tokens have stood
    # for it. That is what makes it the right key for a spend-once ledger.
    nonce: str
    # When the trial was first offered, and so — with TRIAL_LIFE_S — the one
    # deadline no amount of re-signing moves.
    offered: int
    expires: int


def _mint(item_id: int, user_id: int | None, served: Served, nonce: str, offered: int) -> str:
    ttl = TOKEN_TTL_S if user_id else ANON_TOKEN_TTL_S
    # Never past the trial's own deadline, which is what bounds the total life of
    # everything signed for one trial and lets the ledger be sized to cover it.
    expires = min(int(time.time()) + ttl, offered + TRIAL_LIFE_S)
    payload = (
        f"{item_id}.{user_id or 0}.{int(served.repeat)}.{int(served.shared)}"
        f".{nonce}.{offered}.{expires}"
    )
    return f"{payload}.{_sign(payload)}"


def issue(item_id: int, user_id: int | None, served: Served) -> str:
    """A token asserting that `item_id` was offered to `user_id` (0 = nobody yet,
    which is every trial served before its owner has answered anything), and how
    it was offered."""
    # The nonce makes every issuance distinct. Without it, two anonymous callers
    # served the same item in the same second get byte-identical tokens — and a
    # spend-once ledger keyed on them would let the first of them answer and
    # refuse the second, which is two strangers colliding rather than a replay.
    return _mint(item_id, user_id, served, secrets.token_urlsafe(8), int(time.time()))


def _read(token: str | None, item_id: int, user_id: int | None) -> Trial:
    """Raise unless `token` is this server's proof that it offered `item_id` to
    `user_id` and the trial is still live; return what it says. This token's own
    expiry is the caller's to check, being the only thing here a fresh signature
    can fix.

    Every field is signed, so all of this is the server reading back its own
    claim.
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
        offered, expires = int(parts[5]), int(parts[6])
    except ValueError as e:  # we signed it, so this is corruption, not an attack
        raise InvalidTrial("unreadable trial token") from e
    if token_item != item_id:
        raise InvalidTrial("trial token is for a different item")
    if token_user != (user_id or 0):
        raise InvalidTrial("trial token was issued to a different session")
    # Checked here rather than beside the expiry, because it is the deadline
    # `reissue` may not reach past either — and a trial past it is not a stale
    # token but a finished one.
    if time.time() >= offered + TRIAL_LIFE_S:
        raise InvalidTrial("trial is too old to answer")
    return Trial(
        Served(repeat=bool(repeat), shared=bool(shared)), parts[NONCE_FIELD], offered, expires
    )


def redeem(token: str | None, item_id: int, user_id: int | None) -> Trial:
    """`_read`, plus this token's own expiry. Nothing is consumed here —
    spending an anonymous trial is the caller's job, because only it knows
    whether the answer went on to be recorded."""
    trial = _read(token, item_id, user_id)
    if time.time() >= trial.expires:
        raise TrialExpired("trial token has expired")
    return trial


def reissue(token: str | None, item_id: int, user_id: int | None) -> str:
    """The same trial, signed again with a fresh expiry.

    Deliberately incapable of saying anything the old token didn't: a caller
    gets back the trial they were already holding, offered the way it was
    already offered, dying when it was always going to die. So the only clock
    it skips is this token's own — an expired one is the whole reason to call
    it, and a live one re-signs just as happily, which is worth nothing that
    holding it isn't.
    """
    trial = _read(token, item_id, user_id)
    return _mint(item_id, user_id, trial.served, trial.nonce, trial.offered)
