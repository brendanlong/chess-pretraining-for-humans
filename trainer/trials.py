"""Trial tokens: the server's own proof that it offered this item to this caller.

`/api/answer` returns the answer key, so it has to be reachable only by
answering a trial that was really served. Recording that server-side meant a
`users` row had to exist before the first trial — and minting a row on arrival
makes arriving a write, a write needs a limit, and a limit on arriving is a gate
in front of the first trial, which SPEC forbids. A signed token needs no row:
the pending trial becomes state the client holds and cannot forge.

The token names the holder as well as the item. That part is load-bearing: a
token issued to nobody in particular could otherwise be fetched by a throwaway
cookieless client and spent by the signed-in one, which is the pre-commit peek
it exists to prevent. Anonymous tokens (issued before anyone has answered
anything) are interchangeable between anonymous callers, but spending one mints
a fresh row and records the answer *there*, so a peek can never accumulate onto
an identity that keeps its cookie.
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

SECRET_ENV_VAR = "TRIAL_TOKEN_SECRET"


def _load_secret() -> bytes:
    configured = os.environ.get(SECRET_ENV_VAR, "")
    if configured:
        return configured.encode()
    # A laptop or a test run. An ephemeral key behaves exactly like a configured
    # one until the process restarts, which rejects the trials in flight — fine
    # locally, and the log line is here so that it is never quietly the case in
    # production, where `fly secrets set TRIAL_TOKEN_SECRET=…` supplies one.
    log.warning(
        "%s is unset — using an ephemeral key. Trials in flight will be refused across a restart.",
        SECRET_ENV_VAR,
    )
    return secrets.token_bytes(32)


SECRET = _load_secret()


class InvalidTrial(Exception):
    """The token is missing, malformed, forged, expired, or someone else's."""


def _sign(payload: str) -> str:
    mac = hmac.new(SECRET, payload.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(mac).decode().rstrip("=")


def issue(item_id: int, user_id: int | None) -> str:
    """A token asserting that `item_id` was offered to `user_id` (0 = nobody
    yet, which is every trial served before its owner has answered anything)."""
    payload = f"{item_id}.{user_id or 0}.{int(time.time()) + TOKEN_TTL_S}"
    return f"{payload}.{_sign(payload)}"


def redeem(token: str | None, item_id: int, user_id: int | None) -> None:
    """Raise unless `token` is this server's proof that it offered `item_id` to
    `user_id`. Returns nothing: there is no state to spend, which is the point."""
    parts = (token or "").split(".")
    if len(parts) != 4:
        raise InvalidTrial("malformed trial token")
    payload, mac = ".".join(parts[:3]), parts[3]
    # compare_digest before parsing anything: the fields are only meaningful
    # once we know we wrote them.
    if not hmac.compare_digest(mac, _sign(payload)):
        raise InvalidTrial("trial token does not verify")
    try:
        token_item, token_user, expires = (int(p) for p in parts[:3])
    except ValueError as e:  # we signed it, so this is corruption, not an attack
        raise InvalidTrial("unreadable trial token") from e
    if token_item != item_id:
        raise InvalidTrial("trial token is for a different item")
    if token_user != (user_id or 0):
        raise InvalidTrial("trial token was issued to a different session")
    if time.time() >= expires:
        raise InvalidTrial("trial token has expired")
