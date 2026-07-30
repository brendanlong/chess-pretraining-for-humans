"""SQLite storage for items, users, and responses."""

import contextlib
import os
import sqlite3
from pathlib import Path

# In a container the database lives on a mounted volume, not in the checkout,
# and the server has no argv to take a path from.
DEFAULT_DB = Path(os.environ.get("TRAINER_DB", "data/items.db"))
USERS_NAME_INDEX = "idx_users_name_nocase"

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id INTEGER PRIMARY KEY,
    fen TEXT NOT NULL UNIQUE,
    best_uci TEXT NOT NULL,
    distractor_uci TEXT NOT NULL,
    distractor_source TEXT NOT NULL,  -- 'game' (move actually played) or 'multipv'
    cp_best INTEGER,                  -- mover POV, deep search
    mate_best INTEGER,
    cp_distractor INTEGER,
    mate_distractor INTEGER,
    wp_best REAL NOT NULL,
    wp_distractor REAL NOT NULL,
    gap_wp REAL NOT NULL,
    pv_best TEXT,        -- space-separated UCI line starting with best_uci
    pv_distractor TEXT,  -- space-separated UCI line starting with distractor_uci
    learnable INTEGER NOT NULL,       -- shallow search agrees which move is better
    depth_deep INTEGER NOT NULL,
    depth_shallow INTEGER NOT NULL,
    rating REAL NOT NULL,             -- adaptive difficulty rating, seeded from gap_wp
    attempts INTEGER NOT NULL DEFAULT 0,
    correct INTEGER NOT NULL DEFAULT 0,
    ply INTEGER,
    game_url TEXT,
    mover_elo INTEGER,
    time_control TEXT
);

-- A user starts as a guest: a row with no password, reachable only through
-- the session token in its owner's cookie. Signing up sets name/password on
-- that same row, so claiming an account keeps every response already given.
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,  -- chosen username once claimed; opaque 'guest_…' before
    rating REAL NOT NULL DEFAULT 700,
    calib_step REAL NOT NULL DEFAULT 250,  -- staircase step; < ~40 means calibrated
    attempts INTEGER NOT NULL DEFAULT 0,
    password_hash TEXT,  -- NULL means guest
    email TEXT,          -- optional and unverified; only ever used for password reset
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,  -- sha256 of the cookie token; a DB leak grants no logins
    user_id INTEGER NOT NULL REFERENCES users(id),
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    choice_uci TEXT NOT NULL,
    correct INTEGER NOT NULL,
    response_ms INTEGER,
    user_rating_before REAL,
    user_rating_after REAL,
    item_rating_before REAL,
    item_rating_after REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_items_rating ON items(rating);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses(user_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def connect(path: Path = DEFAULT_DB, check_same_thread: bool = True) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=10000")
    user_cols = {row[1] for row in conn.execute("PRAGMA table_info(users)")}
    if "calib_step" not in user_cols:
        conn.execute("ALTER TABLE users ADD COLUMN calib_step REAL NOT NULL DEFAULT 250")
    for col, decl in (
        ("password_hash", "TEXT"),
        ("email", "TEXT"),
        # No datetime('now') default: SQLite rejects non-constant defaults in
        # ALTER TABLE, and pre-auth rows have no signup date to record anyway.
        ("created_at", "TEXT"),
    ):
        if col not in user_cols:
            conn.execute(f"ALTER TABLE users ADD COLUMN {col} {decl}")
    # Each migration below is guarded by a read: connecting to an already-current
    # database must not take a write lock, or opening the server while the
    # labeler holds a transaction fails instead of just working.
    if conn.execute("SELECT 1 FROM users WHERE created_at IS NULL LIMIT 1").fetchone():
        # Rows from before accounts have no signup date; give them one so the
        # column means the same thing everywhere (the guest sweep reads it).
        conn.execute("UPDATE users SET created_at = datetime('now') WHERE created_at IS NULL")
    # Usernames are compared case-insensitively, so the constraint has to be
    # too — otherwise 'Bob' and 'bob' both fit and lookups pick one at random.
    # It's an index over a check the queries make anyway, so a database that
    # can't take it — legacy names already collide, or someone else holds the
    # write lock — carries on without it rather than failing to open.
    if not any(row[1] == USERS_NAME_INDEX for row in conn.execute("PRAGMA index_list(users)")):
        with contextlib.suppress(sqlite3.Error):
            conn.execute(f"CREATE UNIQUE INDEX {USERS_NAME_INDEX} ON users(name COLLATE NOCASE)")
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    for col in ("pv_best", "pv_distractor"):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")
    conn.commit()  # migrations include a write; don't leave the file locked
    return conn
