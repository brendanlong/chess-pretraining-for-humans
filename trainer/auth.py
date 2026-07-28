"""Guest identities, optional accounts, and the sessions behind both.

The app must be answerable within seconds of landing, so identity is
anonymous-first: the first request mints a guest `users` row and hands back
an opaque session token. Signing up attaches a username and password to that
same row, so an account is a claim on history already earned rather than a
gate in front of it.

Nothing here touches the trial flow; ratings and responses are keyed on
`users.id` exactly as before.
"""

import hashlib
import re
import secrets
import sqlite3
import time

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

COOKIE_NAME = "sid"
SESSION_DAYS = 365
# Guest rows carry a random name so nothing about them is guessable, and the
# prefix is reserved so a signup can never collide with one.
GUEST_PREFIX = "guest_"
USERNAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,31}$")
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_LEN = 200  # argon2 is happy to hash megabytes; we shouldn't let it

_hasher = PasswordHasher()


class AuthError(Exception):
    """A user-facing auth failure (bad input, taken name, wrong password)."""


class RateLimited(AuthError):
    """Too many signup/login attempts from one address."""


# --- credentials ----------------------------------------------------------


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password_hash: str | None, password: str) -> bool:
    if not password_hash:
        return False
    try:
        return _hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHashError):
        return False


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


def claim(
    conn: sqlite3.Connection, user: dict, username: str, password: str, email: str | None
) -> dict:
    """Attach credentials to an existing (guest) row. Nothing else changes:
    rating, calibration state and responses carry over untouched."""
    if not is_guest(user):
        raise AuthError("This session is already signed in.")
    username = check_username(username)
    check_password(password)
    email = check_email(email)
    if find_by_username(conn, username):
        raise AuthError("That username is taken.")
    conn.execute(
        "UPDATE users SET name = ?, password_hash = ?, email = ? WHERE id = ?",
        (username, hash_password(password), email, user["id"]),
    )
    conn.commit()
    return get_user(conn, user["id"])


def authenticate(conn: sqlite3.Connection, username: str, password: str) -> dict:
    user = find_by_username(conn, username.strip())
    # Hash anyway when the name is unknown so a miss costs the same as a wrong
    # password — otherwise timing enumerates usernames.
    if user is None or is_guest(user):
        _hasher.hash(password[:MAX_PASSWORD_LEN])
        raise AuthError("Wrong username or password.")
    if not verify_password(user["password_hash"], password):
        raise AuthError("Wrong username or password.")
    return user


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


# --- rate limiting --------------------------------------------------------


class RateLimiter:
    """Fixed-window per-key counter, in memory.

    Deliberately not a captcha: this is a small app, and the cost of a wrong
    guess here should be a wait, not a lost signup. State is per-process and
    resets on restart, which is fine for the threat (casual scripted abuse).
    """

    def __init__(self, limit: int, window_s: float):
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, list[float]] = {}

    def check(self, key: str, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        hits = [t for t in self._hits.get(key, []) if now - t < self.window_s]
        if len(hits) >= self.limit:
            raise RateLimited("Too many attempts. Wait a few minutes and try again.")
        hits.append(now)
        self._hits[key] = hits
        if len(self._hits) > 10_000:  # crude bound; oldest keys are cheapest to lose
            for k in [k for k, v in self._hits.items() if not v or now - v[-1] > self.window_s]:
                del self._hits[k]
