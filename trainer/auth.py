"""Guest identities, optional accounts, and the sessions behind both.

The app must be answerable within seconds of landing, so identity is
anonymous-first: the first request mints a guest `users` row and hands back
an opaque session token. Signing up attaches a username and password to that
same row, so an account is a claim on history already earned rather than a
gate in front of it.

Nothing here touches the trial flow; ratings and responses are keyed on
`users.id` exactly as before.
"""

import contextlib
import hashlib
import re
import secrets
import sqlite3
import threading
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

COOKIE_NAME = "sid"
SESSION_DAYS = 365
# A second, absolute bound. The sliding window above is refreshed on every
# request, so a token that is used once a year never expires at all — a stolen
# cookie is a permanent credential unless something ends it. This caps the
# total life of one token regardless of use; it's generous because the stakes
# are a chess rating, but "generous" and "unbounded" are different claims.
SESSION_MAX_DAYS = 730
# A row that answered nothing and whose session has gone cold is noise, not
# history, so it is swept. Anything with a response or a password is never
# touched. Answering is what creates a row now, so this mostly reclaims two
# things: the guests that arrival-minting left behind, and the gap between
# minting an identity and recording the answer that earned it.
#
# Comfortably longer than the hour `session_user` waits before refreshing
# `last_seen`, because a session in continuous use can look that stale — and
# sweeping a live identity would silently re-identify its owner, losing whatever
# rating they had climbed to.
GUEST_TTL_HOURS = 3
# Guest rows carry a random name so nothing about them is guessable, and the
# prefix is reserved so a signup can never collide with one.
GUEST_PREFIX = "guest_"
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 200  # argon2 is happy to hash megabytes; we shouldn't let it

_hasher = PasswordHasher()
# Verified against when the username is unknown, so a miss costs the same as a
# wrong password and timing can't enumerate accounts.
_DUMMY_HASH = _hasher.hash("no such user")
# argon2 is memory-hard on purpose — 64 MiB per hash at the default settings —
# and endpoints don't serialize on the database lock, so without a cap a burst
# of concurrent signups would each claim their own 64 MiB and take the box out.
# The short wait is what keeps this from trading the out-of-memory for a
# stalled thread pool: sync endpoints run on a fixed pool, so a caller merely
# waiting still holds a thread the trial flow needs. Past the wait we shed.
#
# Sized as a small multiple of one hash (~50ms here): long enough that real
# contention resolves rather than 503s, short enough that a flood can only
# hold a thread briefly. Measured under 300 concurrent logins from distinct
# addresses, the trial flow's worst case goes 7.7s uncapped -> 0.55s.
HASH_CONCURRENCY = 4
HASH_WAIT_S = 0.1
_hash_slots = threading.Semaphore(HASH_CONCURRENCY)


class AuthError(Exception):
    """A user-facing auth failure (bad input, taken name, wrong password)."""


class RateLimited(AuthError):
    """Too many signup/login attempts from one address."""


class AuthBusy(AuthError):
    """Password hashing is saturated; the client should retry shortly."""


BUSY_MESSAGE = "Too many people signing in right now. Try again in a moment."


@contextlib.contextmanager
def _hash_slot():
    if not _hash_slots.acquire(timeout=HASH_WAIT_S):
        raise AuthBusy(BUSY_MESSAGE)
    try:
        yield
    finally:
        _hash_slots.release()


# --- credentials ----------------------------------------------------------


def hash_password(password: str) -> str:
    with _hash_slot():
        return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    """Constant-ish work whether or not there is a hash to check against."""
    if len(password) > MAX_PASSWORD_LEN:
        return False  # nothing this long was ever hashed; don't pay to find out
    with _hash_slot():  # AuthBusy propagates: a saturated box says so
        try:
            verified = _hasher.verify(password_hash or _DUMMY_HASH, password)
        except (VerifyMismatchError, VerificationError, InvalidHashError):
            return False
    return verified and bool(password_hash)


def check_username(name: str) -> str:
    name = name.strip()
    if not USERNAME_RE.match(name):
        raise AuthError(
            "Username must be 3 to 32 characters: letters, digits, and . _ - "
            "(starting with a letter or digit)."
        )
    if name.lower().startswith(GUEST_PREFIX):
        raise AuthError(f"Usernames can't start with '{GUEST_PREFIX}'.")
    return name


def check_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LEN:
        raise AuthError(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if len(password) > MAX_PASSWORD_LEN:
        raise AuthError(f"Password must be at most {MAX_PASSWORD_LEN} characters.")
    return password


def check_email(email: str | None) -> str | None:
    """Email is optional and never verified — its only job is password reset,
    so the check is just 'this could plausibly be delivered to'."""
    email = (email or "").strip()
    if not email:
        return None
    if len(email) > 254 or not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise AuthError("That doesn't look like an email address.")
    return email


# --- users ----------------------------------------------------------------


def create_guest(conn: sqlite3.Connection, start_rating: float, calib_step: float) -> dict:
    name = GUEST_PREFIX + secrets.token_hex(8)
    cur = conn.execute(
        """INSERT INTO users (name, rating, calib_step, created_at)
           VALUES (?, ?, ?, datetime('now'))""",
        (name, start_rating, calib_step),
    )
    conn.commit()
    return get_user(conn, cur.lastrowid)  # pyright: ignore[reportArgumentType]


def get_user(conn: sqlite3.Connection, user_id: int) -> dict:
    row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if row is None:
        raise AuthError("no such user")
    return dict(row)


def is_guest(user: dict) -> bool:
    return not user["password_hash"]


def display_name(user: dict) -> str | None:
    """What the header chip shows — guests have no name worth showing."""
    return None if is_guest(user) else user["name"]


def find_by_username(conn: sqlite3.Connection, name: str) -> dict | None:
    """The row a typed username refers to, or None.

    Raises rather than picking one when two rows answer to the same name. That
    only happens on a database whose case-insensitive unique index couldn't be
    created (see db.connect, which logs it), and picking silently is worse than
    failing: `trainer.account set-password kim` resolving to `Kim`'s row hands
    one user another's account and history.
    """
    rows = conn.execute(
        "SELECT * FROM users WHERE name = ? COLLATE NOCASE ORDER BY id", (name,)
    ).fetchall()
    if len(rows) > 1:
        raise AuthError(
            f"{name!r} matches {len(rows)} rows case-insensitively; "
            "the collision has to be resolved before this name can be used."
        )
    return dict(rows[0]) if rows else None


def validate_signup(username: str, password: str, email: str | None) -> tuple[str, str | None]:
    """Everything checkable without touching the database, so the caller can
    reject typos before paying for an argon2 hash."""
    username = check_username(username)
    check_password(password)
    return username, check_email(email)


def check_name_free(conn: sqlite3.Connection, username: str) -> None:
    if find_by_username(conn, username):
        raise AuthError("That username is taken.")


def create_account(
    conn: sqlite3.Connection,
    username: str,
    password_hash: str,
    email: str | None,
    start_rating: float,
    calib_step: float,
) -> dict:
    """Sign up with no history to claim.

    Nothing writes a `users` row until the first answer, so someone who opens
    the drawer before answering anything has no guest row for `claim` to attach
    credentials to. Same row shape either way — this one just starts empty.
    """
    check_name_free(conn, username)
    cur = conn.execute(
        """INSERT INTO users (name, rating, calib_step, password_hash, email, created_at)
           VALUES (?, ?, ?, ?, ?, datetime('now'))""",
        (username, start_rating, calib_step, password_hash, email),
    )
    conn.commit()
    return get_user(conn, cur.lastrowid)  # pyright: ignore[reportArgumentType]


def check_claimable(conn: sqlite3.Connection, user_id: int, username: str) -> None:
    """The database half of signup validation, cheap enough to run before the
    password hash — otherwise a taken name costs an argon2 each time it's
    probed, which is both an enumeration oracle and free work for an attacker.
    `claim` repeats it under the lock that writes, where it decides the race."""
    if not is_guest(get_user(conn, user_id)):
        raise AuthError("This session is already signed in.")
    check_name_free(conn, username)


def claim(
    conn: sqlite3.Connection, user_id: int, username: str, password_hash: str, email: str | None
) -> dict:
    """Attach credentials to an existing (guest) row. Nothing else changes:
    rating, calibration state and responses carry over untouched.

    Takes an already-computed hash so the caller can do the slow part without
    holding the database lock; the taken-name check and the write stay
    together here, under whatever lock the caller holds."""
    check_claimable(conn, user_id, username)
    conn.execute(
        "UPDATE users SET name = ?, password_hash = ?, email = ? WHERE id = ?",
        (username, password_hash, email, user_id),
    )
    conn.commit()
    return get_user(conn, user_id)


def delete_user(conn: sqlite3.Connection, user_id: int) -> dict[str, int]:
    """Erase a user row and everything that points at it. Returns row counts.

    The one place in the app that destroys research data on purpose. The
    privacy policy promises that deleting an account takes its responses with
    it, and keeping the answers while dropping the name would leave us holding
    data a user believes is gone — cheaper to lose the rows than to redefine
    the word.

    `sessions` and `responses` both reference `users(id)`, so the row itself
    goes last, and all of it goes in one transaction: a half-deleted account
    is a live session pointing at nothing.
    """
    with conn:  # commits on success, rolls the whole thing back on failure
        counts = {
            "responses": conn.execute(
                "DELETE FROM responses WHERE user_id = ?", (user_id,)
            ).rowcount,
            "sessions": conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,)).rowcount,
        }
        counts["users"] = conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount
    return counts


def credential_for(user: dict | None) -> str | None:
    """The hash to check a login against — None for unknown or guest rows,
    which `verify_password` still spends a full verify on."""
    return None if user is None or is_guest(user) else user["password_hash"]


# --- sessions -------------------------------------------------------------


def _token_hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def start_session(conn: sqlite3.Connection, user_id: int) -> str:
    """Returns the raw token; only its hash is stored."""
    token = secrets.token_urlsafe(32)
    conn.execute(
        "INSERT INTO sessions (token_hash, user_id) VALUES (?, ?)",
        (_token_hash(token), user_id),
    )
    conn.commit()
    return token


def session_user(conn: sqlite3.Connection, token: str | None) -> dict | None:
    if not token:
        return None
    th = _token_hash(token)
    row = conn.execute(
        f"""SELECT users.* FROM sessions JOIN users ON users.id = sessions.user_id
            WHERE sessions.token_hash = ?
              AND sessions.last_seen > datetime('now', '-{SESSION_DAYS} days')
              AND sessions.created_at > datetime('now', '-{SESSION_MAX_DAYS} days')""",
        (th,),
    ).fetchone()
    if row is None:
        return None
    # Keep the sliding expiry fresh without writing on every single request.
    conn.execute(
        "UPDATE sessions SET last_seen = datetime('now') WHERE token_hash = ?"
        " AND last_seen < datetime('now', '-1 hour')",
        (th,),
    )
    conn.commit()
    return dict(row)


def end_session(conn: sqlite3.Connection, token: str | None) -> None:
    if token:
        conn.execute("DELETE FROM sessions WHERE token_hash = ?", (_token_hash(token),))
        conn.commit()


def revoke_sessions(conn: sqlite3.Connection, user_id: int) -> int:
    """Sign a user out everywhere. Returns the number of sessions dropped.

    What a password change is *for*, when the reason for it is that someone
    else knows the old one. Rotating the hash alone stops them signing in
    again while leaving the session they already have working, which makes the
    only recovery path this app has (`trainer.account set-password`, since no
    reset email exists yet) unable to actually recover anything.
    """
    cur = conn.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    conn.commit()
    return cur.rowcount


def sweep(conn: sqlite3.Connection) -> None:
    """Drop dead sessions and the empty guest rows they leave behind.

    Arriving is enough to mint a guest, so crawlers and health checks would
    otherwise grow `users` without bound. Only rows with no password, no
    trials, no responses and no warm session are touched — the experimental
    record is never in reach.
    """
    conn.execute(
        "DELETE FROM sessions WHERE last_seen < datetime('now', ?)", (f"-{SESSION_DAYS} days",)
    )
    conn.execute(
        """DELETE FROM users
           WHERE password_hash IS NULL
             AND attempts = 0
             AND created_at IS NOT NULL
             AND created_at < datetime('now', ?)
             AND NOT EXISTS (SELECT 1 FROM responses WHERE responses.user_id = users.id)
             AND NOT EXISTS (SELECT 1 FROM sessions
                             WHERE sessions.user_id = users.id
                               AND sessions.last_seen > datetime('now', ?))""",
        (f"-{GUEST_TTL_HOURS} hours", f"-{GUEST_TTL_HOURS} hours"),
    )
    conn.execute("DELETE FROM sessions WHERE user_id NOT IN (SELECT id FROM users)")
    conn.commit()


# --- rate limiting --------------------------------------------------------


class RateLimiter:
    """Sliding-window per-key counter, in memory.

    Deliberately not a captcha: this is a small app, and the cost of a wrong
    guess here should be a wait, not a lost signup. State is per-process and
    resets on restart, which is fine for the threat (casual scripted abuse).

    `consume` is the whole interface: one atomic check-and-record, taken
    *before* the work it rations. Checking first and recording once the
    outcome is known reads a counter that nothing in flight has incremented
    yet, so a concurrent burst all passes at once.

    Nothing is ever handed back. Refunding slots for outcomes that "shouldn't
    count" is where every version of this file went wrong: it needs an exit
    path to be right on, and it never was. A limit loose enough that a
    fumbled form can't reach it buys the same forgiveness for free.
    """

    MAX_KEYS = 10_000
    DEFAULT_MESSAGE = "Too many attempts. Wait a few minutes and try again."

    def __init__(self, limit: int, window_s: float, message: str = DEFAULT_MESSAGE):
        self.limit = limit
        self.window_s = window_s
        # The default speaks to someone who typed something wrong. Not every
        # limiter rations a guess: the one in front of answering is refusing a
        # perfectly good answer because the address is busy, and the spend-once
        # ledger isn't rationing anything at all. Telling either of them they
        # have made too many attempts is untrue and unhelpful.
        self.message = message
        self._hits: dict[str, list[float]] = {}
        self._lock = threading.Lock()  # endpoints no longer serialize elsewhere

    def _live(self, key: str, now: float) -> list[float]:
        return [t for t in self._hits.get(key, []) if now - t < self.window_s]

    def _prune(self, now: float) -> None:
        """Bound the key space. Runs only when over the cap, and cuts well
        under it, so the scan amortizes instead of repeating every call."""
        if len(self._hits) <= self.MAX_KEYS:
            return
        for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window_s]:
            del self._hits[key]
        if len(self._hits) > self.MAX_KEYS:
            # Still over, so the traffic is spread across more keys than we
            # will track and something has to be forgiven early. Forgive the
            # keys with the *fewest* live hits, not the least recently active
            # ones. Keys are cheap for an attacker to manufacture — a login key
            # is whatever username they typed — and evicting by age lets a
            # flood of one-hit junk push out the nearly-exhausted counter of
            # the account they are actually guessing at, handing themselves a
            # fresh budget. Fewest-first inverts that: evicting a key with N
            # hits means spending N requests on each of thousands of decoys,
            # which is what the per-address budget in front makes expensive.
            ranked = sorted(self._hits, key=lambda k: (len(self._live(k, now)), self._hits[k][-1]))
            for key in ranked[: len(self._hits) - self.MAX_KEYS * 3 // 4]:
                del self._hits[key]

    def _record(self, key: str, now: float) -> None:
        self._hits[key] = [*self._live(key, now), now]
        self._prune(now)

    def consume(self, key: str, now: float | None = None) -> None:
        """Spend a slot, or raise."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._live(key, now)) >= self.limit:
                raise RateLimited(self.message)
            self._record(key, now)

    def clear(self, key: str) -> None:
        """Forget a key entirely.

        Unlike a per-request refund — which needs every exit path to be right,
        and is where earlier versions of this went wrong — this is reachable
        only by proving you know the password. An attacker who doesn't can
        never trigger it, and forgetting to call it just leaves someone
        slightly more throttled than intended.
        """
        with self._lock:
            self._hits.pop(key, None)
