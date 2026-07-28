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
# Anyone can mint a guest just by arriving, so untouched ones are swept: a
# guest that answered nothing and whose session has gone cold is indexable
# noise, not history. Anything with a response or a password is never touched.
GUEST_TTL_DAYS = 1
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
# and endpoints no longer serialize on the database lock, so without a cap a
# burst of concurrent signups would each claim their own 64 MiB and take the
# box out. Two limits, because bounding memory alone isn't enough: the server
# runs sync endpoints on a fixed thread pool, so callers merely *waiting* for a
# hash still occupy threads the trial flow needs. HASH_QUEUE keeps auth's share
# of that pool small and sheds anything beyond it immediately.
HASH_CONCURRENCY = 4  # hashing at once — bounds memory
HASH_QUEUE = 12  # inside the hasher at all, waiting included — bounds threads
HASH_WAIT_S = 1.5
_hash_slots = threading.Semaphore(HASH_CONCURRENCY)
_hash_queue = threading.Semaphore(HASH_QUEUE)


class AuthError(Exception):
    """A user-facing auth failure (bad input, taken name, wrong password)."""


class RateLimited(AuthError):
    """Too many signup/login attempts from one address."""


class AuthBusy(AuthError):
    """Password hashing is saturated; the client should retry shortly."""


BUSY_MESSAGE = "Too many people signing in right now. Try again in a moment."


@contextlib.contextmanager
def _hash_slot():
    if not _hash_queue.acquire(blocking=False):
        raise AuthBusy(BUSY_MESSAGE)  # don't even hold a thread to wait
    try:
        if not _hash_slots.acquire(timeout=HASH_WAIT_S):
            raise AuthBusy(BUSY_MESSAGE)
        try:
            yield
        finally:
            _hash_slots.release()
    finally:
        _hash_queue.release()


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
    row = conn.execute("SELECT * FROM users WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
    return dict(row) if row else None


def validate_signup(username: str, password: str, email: str | None) -> tuple[str, str | None]:
    """Everything checkable without touching the database, so the caller can
    reject typos before paying for an argon2 hash."""
    username = check_username(username)
    check_password(password)
    return username, check_email(email)


def check_claimable(conn: sqlite3.Connection, user_id: int, username: str) -> None:
    """The database half of signup validation, cheap enough to run before the
    password hash — otherwise a taken name costs an argon2 each time it's
    probed, which is both an enumeration oracle and free work for an attacker.
    `claim` repeats it under the lock that writes, where it decides the race."""
    if not is_guest(get_user(conn, user_id)):
        raise AuthError("This session is already signed in.")
    if find_by_username(conn, username):
        raise AuthError("That username is taken.")


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
              AND sessions.last_seen > datetime('now', '-{SESSION_DAYS} days')""",
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
        (f"-{GUEST_TTL_DAYS} days", f"-{GUEST_TTL_DAYS} days"),
    )
    conn.execute("DELETE FROM sessions WHERE user_id NOT IN (SELECT id FROM users)")
    conn.commit()


# --- rate limiting --------------------------------------------------------


class RateLimiter:
    """Sliding-window per-key counter, in memory.

    Deliberately not a captcha: this is a small app, and the cost of a wrong
    guess here should be a wait, not a lost signup. State is per-process and
    resets on restart, which is fine for the threat (casual scripted abuse).

    The only gate is `consume`: one atomic check-and-record, taken *before*
    the slow work. Checking first and recording after the outcome is known
    reads a counter that nothing in flight has incremented yet, so a
    concurrent burst all passes at once. Where an outcome shouldn't have
    counted after all — a login that turned out to be correct — the slot is
    handed back with `release` rather than never taken.
    """

    MAX_KEYS = 10_000

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
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
            # Still over, so the traffic is spread across more addresses than
            # we will track. Forget the least recently active: dropping a key
            # forgives an address early, it can never block a legitimate one.
            by_age = sorted(self._hits, key=lambda k: self._hits[k][-1])
            for key in by_age[: len(self._hits) - self.MAX_KEYS * 3 // 4]:
                del self._hits[key]

    def _record(self, key: str, now: float) -> None:
        self._hits[key] = [*self._live(key, now), now]
        self._prune(now)

    def consume(self, key: str, now: float | None = None) -> None:
        """Take a slot, or raise. Held for the duration of the work."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if len(self._live(key, now)) >= self.limit:
                raise RateLimited("Too many attempts. Wait a few minutes and try again.")
            self._record(key, now)

    def release(self, key: str) -> None:
        """Give a slot back once the outcome turns out not to be worth
        counting.

        This drops the key's last recorded entry, which under concurrency may
        belong to a different in-flight request rather than the caller. That
        is safe *because at most one release follows each successful consume*:
        what bounds a key is the count of live entries, and the count doesn't
        care which one is dropped. Callers must keep that one-to-one — both
        here take their slot outside the `try` whose `finally` releases it.
        Release more than you consumed and you take a live request's slot.

        Do not read this as "each caller gets its own slot back"; nothing
        tracks callers. The only visible effect is that the surviving entry is
        the older timestamp, so the window rolls over a request-duration
        sooner — always in the forgiving direction.
        """
        with self._lock:
            hits = self._hits.get(key)
            if hits:
                hits.pop()
