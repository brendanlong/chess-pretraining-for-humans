"""SQLite storage for items, users, and responses."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Protocol

from . import rating
from .rating import difficulty_rating, shallow_gap_of

log = logging.getLogger(__name__)


class Queryable(Protocol):
    """Anything that runs a statement: a connection, or a transaction handle.

    What the storage helpers ask for, so a caller inside a transaction can hand
    them something with no way to end it and a script can hand them a connection.
    """

    def execute(self, sql: str, parameters=(), /) -> sqlite3.Cursor: ...


def _schema_version(conn: sqlite3.Connection) -> int:
    """0 for a database written before `meta` existed.

    An unreadable value is treated as newer than us rather than older: the
    migrations it gates are one-way, and running one against a file we can't
    identify is the failure that isn't recoverable.
    """
    row = conn.execute("SELECT value FROM meta WHERE key = 'schema_version'").fetchone()
    if row is None:
        return 0
    try:
        return int(row["value"])
    except ValueError:
        log.error("meta.schema_version is %r, not a number — skipping migrations", row["value"])
        return SCHEMA_VERSION


# In a container the database lives on a mounted volume, not in the checkout,
# and the server has no argv to take a path from.
DEFAULT_DB = Path(os.environ.get("TRAINER_DB", "data/items.db"))
# How long a statement waits for a lock someone else holds. Generous because the
# writer it waits on may be a bank refresh merging into the live database, and
# failing an answer because an operator was mid-runbook is worse than a pause.
BUSY_TIMEOUT_MS = 10_000
USERS_NAME_INDEX = "idx_users_name_nocase"
# Bumped only for migrations that can't tell from the data whether they ran.
SCHEMA_VERSION = 2

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
    -- Plies of lookahead the comparison needs: the shallowest search that gets
    -- the two moves the right way round and is not contradicted deeper.
    --
    -- Three states, and keeping them apart is what stops `trainer.push_items`
    -- and `trainer.backfill_depth` from disagreeing about which rows still need
    -- the ladder run over them. 1..depth_deep is a depth; 0 is the ladder's
    -- other answer, that not even depth_deep settles it; NULL is that nobody has
    -- asked, which is only ever a row labeled before this was measured, and
    -- reads as "adds nothing" so such a row keeps the difficulty it already had.
    solution_depth INTEGER,
    -- The measurement `solution_depth` summarises: the win-probability gap the
    -- pair was ranked at, at every depth from 1 up, space-separated. Kept
    -- because it is the expensive half — a deep search per item — while every
    -- way of reading it is cheap and none of them is settled. A better
    -- difficulty model can be fitted off these columns without going near
    -- Stockfish again, which is the difference between an afternoon and a
    -- re-label. A gap here can be negative: at that depth the search preferred
    -- the wrong move, which is more than "didn't see it".
    gap_ladder TEXT,
    -- The mean of the ladder's first `label.SHALLOW_PLIES` rungs: the gap as
    -- the shallow end of the search saw it, which is what difficulty is a
    -- function of. Negative where the surface recommends the wrong move.
    shallow_gap REAL,
    learnable INTEGER NOT NULL,       -- solution_depth != 0, once it is measured
    depth_deep INTEGER NOT NULL,
    rating REAL NOT NULL,             -- difficulty: rating.difficulty_rating(shallow_gap)
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
    rating REAL NOT NULL DEFAULT 850,  -- rating.USER_START; every insert sets it explicitly
    calib_step REAL NOT NULL DEFAULT 250,  -- staircase step; < ~40 means calibrated
    attempts INTEGER NOT NULL DEFAULT 0,
    password_hash TEXT,  -- NULL means guest
    email TEXT,          -- optional and unverified; only ever used for password reset
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash TEXT PRIMARY KEY,  -- sha256 of the cookie token; a DB leak grants no logins
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    last_seen TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY,
    -- No ON DELETE CASCADE here, deliberately: responses are the experimental
    -- record, so erasing them must be an explicit act (auth.delete_user),
    -- never the quiet side effect of a user row going away.
    user_id INTEGER NOT NULL REFERENCES users(id),
    item_id INTEGER NOT NULL REFERENCES items(id),
    choice_uci TEXT NOT NULL,
    correct INTEGER NOT NULL,
    response_ms INTEGER,
    user_rating_before REAL,
    user_rating_after REAL,
    -- The item's difficulty as served. Recorded rather than joined so a row
    -- stays interpretable if difficulty_rating's constants are ever retuned.
    item_rating_before REAL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Migrations that can't tell from the data whether they already ran need
-- somewhere to say so. Everything else here is guarded by a read of the thing
-- it changes, which is cheaper and can't get out of step with reality.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_items_rating ON items(rating);
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses(user_id, id);
-- Both response indexes earn their keep. The one above serves "this user's
-- answers, in order"; this one serves "has this user answered this item" — the
-- repeat probe, the unseen-item filters, and /api/stats' first-exposure filter.
-- That last asks it per response, so without `item_id` indexed it walks the
-- user's whole history per row: 700ms at 5k answers.
CREATE INDEX IF NOT EXISTS idx_responses_item ON responses(user_id, item_id, id);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
"""


def open_connection(
    path: Path = DEFAULT_DB, check_same_thread: bool = True, explicit_transactions: bool = False
) -> sqlite3.Connection:
    """A connection to an already-migrated database: settings only, no schema.

    Separate because SQLite scopes these to the connection, not the file, and a
    server opens one per thread — only the first has business migrating.

    `explicit_transactions` turns off the driver's implicit ones, so a
    transaction is a deliberate act with one owner (`server.writing`) rather
    than whatever the first `commit()` happens to end. The offline scripts want
    the default, where `commit()` per batch is the right idiom.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        path,
        check_same_thread=check_same_thread,
        isolation_level=None if explicit_transactions else "",  # pyright: ignore[reportArgumentType]
    )
    conn.row_factory = sqlite3.Row
    # Enforced, not just declared: SQLite ships with foreign keys off, and the
    # setting is per-connection. On, a session or response can never point at a
    # user that isn't there — deletion still has to order its statements, but
    # getting the order wrong is now an error instead of an orphan. Rows that
    # violated the constraint before it was enforced are tolerated (SQLite only
    # checks writes) and age out through the session sweep.
    conn.execute("PRAGMA foreign_keys=ON")
    # Set before anything that might write, because this is the timeout a write
    # has to wait out when someone else holds the lock — the labeler, or another
    # request's transaction. Set afterwards it would apply to everything except
    # the statements that most need it, which would get Python's shorter default.
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    return conn


def connect(path: Path = DEFAULT_DB, check_same_thread: bool = True) -> sqlite3.Connection:
    """Open a database and bring its schema up to date."""
    conn = open_connection(path, check_same_thread)
    conn.executescript(SCHEMA)
    conn.execute("PRAGMA journal_mode=WAL")
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
    #
    # But it carries on *loudly*: without the index two rows can share a name
    # case-insensitively, which is the collision `find_by_username` then refuses
    # to guess at. Failing silently here is what turns a legacy-data wrinkle
    # into an operator handing one user another's account.
    if not any(row[1] == USERS_NAME_INDEX for row in conn.execute("PRAGMA index_list(users)")):
        try:
            conn.execute(f"CREATE UNIQUE INDEX {USERS_NAME_INDEX} ON users(name COLLATE NOCASE)")
        except sqlite3.Error as e:
            log.error(
                "could not create %s (%s) — usernames are NOT uniquely constrained "
                "case-insensitively. Two rows may share a name; `trainer.account` will "
                "refuse to act on one until the collision is resolved by hand.",
                USERS_NAME_INDEX,
                e,
            )
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    for col in ("pv_best", "pv_distractor"):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} TEXT")
    # Rows labeled before lookahead depth was measured keep a NULL here, which
    # `difficulty_rating` reads as "gap only" — so they stay exactly as hard as
    # they were until `trainer.backfill_depth` measures them.
    for col, decl in (
        ("solution_depth", "INTEGER"),
        ("gap_ladder", "TEXT"),
        ("shallow_gap", "REAL"),
    ):
        if col not in item_cols:
            conn.execute(f"ALTER TABLE items ADD COLUMN {col} {decl}")
    # Difficulty is a function of the item alone, so the columns that existed to
    # carry answers back into it go. `items.attempts`/`correct` were a global
    # tally no query reads now. `responses.item_rating_after` recorded a move
    # that doesn't happen; recovering one from an old row needs the K-factor and
    # the branch that froze it during calibration, neither of which exists here,
    # so `deploy/README.md` says to copy it out first.
    #
    # `items.depth_shallow` goes for a different reason: there is no shallow
    # pass any more. The ladder runs to `depth_deep` and grades what it finds
    # instead of rejecting it, so the second depth recorded a cutoff that no
    # longer exists — and a column naming a rule the code doesn't follow is
    # worse than no column.
    for col in ("attempts", "correct", "depth_shallow"):
        if col in item_cols:
            conn.execute(f"ALTER TABLE items DROP COLUMN {col}")
    response_cols = {row[1] for row in conn.execute("PRAGMA table_info(responses)")}
    if "item_rating_after" in response_cols:
        conn.execute("ALTER TABLE responses DROP COLUMN item_rating_after")
    # Registering the function rather than repeating the formula in SQL keeps
    # one definition of difficulty. `push_items` calls it by this name too, from
    # a merge that has to land a new difficulty without waiting for a restart.
    conn.create_function("difficulty_rating", 1, difficulty_rating, deterministic=True)
    conn.create_function("shallow_gap_of", 1, shallow_gap_of, deterministic=True)
    conn.create_function("regraded_user_rating", 2, rating.regraded_user_rating, deterministic=True)
    # Derived from the ladder rather than re-measured: the search is the
    # expensive half and it is already in the row, so a bank that has ladders
    # gains the column the app selects on without going near Stockfish.
    if conn.execute(
        "SELECT 1 FROM items WHERE shallow_gap IS NULL AND gap_ladder IS NOT NULL"
        "   AND gap_ladder != '' LIMIT 1"
    ).fetchone():
        conn.execute(
            "UPDATE items SET shallow_gap = shallow_gap_of(gap_ladder)"
            " WHERE shallow_gap IS NULL AND gap_ladder IS NOT NULL AND gap_ladder != ''"
        )
    # Item difficulty is re-derived, because "a pure function of `gap_wp` and
    # `solution_depth`" has to be true of the rows, not just of the code that
    # writes new ones. Users are
    # regraded in the same transaction, because the two are one change: a rating
    # means nothing except against the difficulties it selects, so moving the
    # items without moving the users would silently re-aim everyone.
    #
    # The item half is guarded by a read of the thing it changes and so is
    # naturally idempotent. The user half can't be — a rating carries no mark of
    # which scale produced it, and a second pass would move someone already
    # correct — so it is gated on `meta`, and the gate is re-read inside
    # `BEGIN IMMEDIATE` because two servers starting together would otherwise
    # both see version 0 and both regrade. The lock is taken only when a read
    # says there is work, so an already-current database still opens without one.
    items_stale = conn.execute(
        "SELECT 1 FROM items"
        " WHERE shallow_gap IS NOT NULL AND rating != difficulty_rating(shallow_gap) LIMIT 1"
    ).fetchone()
    if items_stale or _schema_version(conn) < SCHEMA_VERSION:
        conn.commit()  # release the implicit transaction the ALTERs above opened
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                # Only where there is a gap to derive from. A row the ladder
                # has never run over has no difficulty of its own, so it keeps
                # the one the old curve gave it until a measurement arrives —
                # being stale is a smaller wrong than being invented.
                "UPDATE items SET rating = difficulty_rating(shallow_gap)"
                " WHERE shallow_gap IS NOT NULL AND rating != difficulty_rating(shallow_gap)"
            )
            if (was := _schema_version(conn)) < SCHEMA_VERSION:
                conn.execute("UPDATE users SET rating = regraded_user_rating(rating, ?)", (was,))
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
                    (str(SCHEMA_VERSION),),
                )
                # When, so offline analysis can split `responses` on it: rows on
                # either side carry rating snapshots from different scales, and
                # nothing else in the record says where the boundary is.
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value)"
                    " VALUES ('regraded_at', datetime('now'))"
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    # `sessions` gained ON DELETE CASCADE after databases existed, and SQLite
    # can't add a constraint in place, so an old table is rebuilt once. Orphan
    # rows from before the foreign key was enforced are shed on the way — the
    # constraint would refuse them, and `session_user` could never resolve
    # them anyway.
    fk = conn.execute("PRAGMA foreign_key_list(sessions)").fetchone()
    if fk is not None and fk["on_delete"] != "CASCADE":
        conn.executescript(
            """
            BEGIN;
            CREATE TABLE sessions_cascade (
                token_hash TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_seen TEXT NOT NULL DEFAULT (datetime('now'))
            );
            INSERT INTO sessions_cascade
                SELECT token_hash, user_id, created_at, last_seen FROM sessions
                WHERE user_id IN (SELECT id FROM users);
            DROP TABLE sessions;
            ALTER TABLE sessions_cascade RENAME TO sessions;
            CREATE INDEX idx_sessions_user ON sessions(user_id);
            COMMIT;
            """
        )
    conn.commit()  # migrations include a write; don't leave the file locked
    return conn
