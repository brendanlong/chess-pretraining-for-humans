"""SQLite storage for items, users, and responses."""

import logging
import os
import sqlite3
from pathlib import Path
from typing import Protocol

from .rating import difficulty_rating, shallow_gap_of

log = logging.getLogger(__name__)


class Queryable(Protocol):
    """Anything that runs a statement: a connection, or a transaction handle.

    What the storage helpers ask for, so a caller inside a transaction can hand
    them something with no way to end it and a script can hand them a connection.
    """

    def execute(self, sql: str, parameters=(), /) -> sqlite3.Cursor: ...


# In a container the database lives on a mounted volume, not in the checkout,
# and the server has no argv to take a path from.
DEFAULT_DB = Path(os.environ.get("TRAINER_DB", "data/items.db"))
# How long a statement waits for a lock someone else holds. Generous because the
# writer it waits on may be a bank refresh merging into the live database, and
# failing an answer because an operator was mid-runbook is worse than a pause.
BUSY_TIMEOUT_MS = 10_000
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
    -- The win-probability gap the two moves were ranked at, at every depth from
    -- 1 up, space-separated. Kept whole because it is the expensive half — a
    -- deep search per item — while every way of reading it is cheap and none of
    -- them is settled: a better difficulty model can be fitted off this without
    -- going near Stockfish again, which is the difference between an afternoon
    -- and a re-label. A gap here can be negative, meaning the search at that
    -- depth preferred the wrong move, which is more than "didn't see it".
    gap_ladder TEXT,
    -- The mean of the ladder's first `rating.SHALLOW_PLIES` rungs: the gap as
    -- the shallow end of the search saw it, which is what difficulty is a
    -- function of. Negative where the surface recommends the wrong move.
    shallow_gap REAL,
    -- Whether any depth in the ladder settles the pair the right way round. A
    -- position where none does is one the engine won't hold an answer to, so
    -- there is nothing to teach and it is never served.
    learnable INTEGER NOT NULL,
    rating REAL NOT NULL,             -- difficulty: rating.difficulty_rating(shallow_gap)
    ply INTEGER,
    -- Whether mining took this position through its full gap window or a
    -- narrowed one. The difficulty fit may only use the former: narrowing is
    -- selection on the quantity it regresses, and including the rest moves the
    -- same measurement by a factor of three.
    --
    -- Defaults to 0, which is also what a row of unknown provenance deserves —
    -- "we don't know how this was mined" and "it was aimed at a band" both mean
    -- don't fit on it. So there is nothing here to backfill and no third state
    -- to carry.
    mined_untargeted INTEGER NOT NULL DEFAULT 0,
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
    -- Whether the trial was asked for by item id — somebody's link, followed —
    -- instead of being chosen by adaptive selection. The answer counts like any
    -- other; this is for the analysis, which has to be able to hold out the
    -- rows nobody aimed. 0 for everything the app chose.
    shared INTEGER NOT NULL DEFAULT 0,
    -- Whether the calibration staircase still owned this user's rating when
    -- the answer was scored (`rating.is_calibrating` at that moment). The
    -- staircase moves ratings on rules of its own, so a fit over responses has
    -- to hold those moves out — and inferring them from the size of the rating
    -- delta breaks where a bound clamps the move. The staircase moved this
    -- rating iff calibrating = 1 and shared = 0, since a shared answer during
    -- calibration is scored by Elo (see `server.answer`). NULL on rows from
    -- before the column existed; those still need the delta inference, which
    -- `trainer.fit_anchor` carries.
    calibrating INTEGER,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Facts about the record that aren't derivable from it: `regraded_at`, the
-- moment every user rating moved onto the shallow-gap difficulty scale, and
-- `anchored_at`, the moment item difficulty gained `rating.RESPONSE_ANCHOR` —
-- so that offline analysis can tell which scale a response's rating snapshots
-- came from. Nothing in the app reads either.
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Selection walks ratings outward from a target (`server.pick_item`), over
-- servable items only, which is why the index is partial: it keeps the walk
-- free of entries the WHERE would discard, so what bounds it is how much of
-- the band this user has answered, not how much of the bank is unservable.
CREATE INDEX IF NOT EXISTS idx_items_learnable_rating ON items(rating) WHERE learnable = 1;
CREATE INDEX IF NOT EXISTS idx_responses_user ON responses(user_id, id);
-- Both response indexes earn their keep. The one above serves "this user's
-- answers, in order"; this one serves "has this user answered this item" — the
-- repeat probe, selection's unseen filter, and /api/stats' first-exposure one.
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
        # column means the same thing everywhere.
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
    # A database that predates one of these gets the column; `connect` then
    # refuses to open it until a labeled bank fills the measurement in, because
    # an item with no shallow gap has no difficulty.
    for col, decl in (
        ("gap_ladder", "TEXT"),
        ("shallow_gap", "REAL"),
        ("mined_untargeted", "INTEGER NOT NULL DEFAULT 0"),
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
    # The rest record how a measurement was taken in a form nothing reads:
    # `depth_shallow` a cutoff that isn't in the pipeline, `depth_deep` and
    # `solution_depth` both recoverable from `gap_ladder` — its length, and the
    # deepest run of positive rungs. A column that only restates another is one
    # more thing that can disagree with it.
    for col in ("attempts", "correct", "depth_shallow", "depth_deep", "solution_depth"):
        if col in item_cols:
            conn.execute(f"ALTER TABLE items DROP COLUMN {col}")
    response_cols = {row[1] for row in conn.execute("PRAGMA table_info(responses)")}
    if "item_rating_after" in response_cols:
        conn.execute("ALTER TABLE responses DROP COLUMN item_rating_after")
    # Every row that predates share links was selected by the app, which is what
    # the default says — so there is nothing to backfill.
    if "shared" not in response_cols:
        conn.execute("ALTER TABLE responses ADD COLUMN shared INTEGER NOT NULL DEFAULT 0")
    # No default here, unlike `shared`: whether the staircase owned the rating
    # is not knowable from an old row without the delta inference, and a 0
    # would claim it was. NULL says "recorded before the column existed".
    if "calibrating" not in response_cols:
        conn.execute("ALTER TABLE responses ADD COLUMN calibrating INTEGER")
    # The full rating index is subsumed by the partial one above: every query
    # that ranges or orders on rating also asks for learnable = 1, so all the
    # unfiltered index bought was a second copy of the column to keep current
    # on every bank push. Guarded by a read, like the other migrations.
    if any(row[1] == "idx_items_rating" for row in conn.execute("PRAGMA index_list(items)")):
        conn.execute("DROP INDEX idx_items_rating")
    # Registering the function rather than repeating the formula in SQL keeps
    # one definition of difficulty. `push_items` calls it by this name too, from
    # a merge that has to land a new difficulty without waiting for a restart.
    conn.create_function("difficulty_rating", 1, difficulty_rating, deterministic=True)
    conn.create_function("shallow_gap_of", 1, shallow_gap_of, deterministic=True)
    # One precondition instead of a guard on every read. Every row a labeler
    # writes carries a shallow gap, so a NULL means this database predates the
    # measurement — a restore from far enough back. Saying so once, plainly, is
    # better than degrading quietly everywhere that reads it.
    if conn.execute("SELECT 1 FROM items WHERE shallow_gap IS NULL LIMIT 1").fetchone():
        raise RuntimeError(
            f"{path} has items with no shallow_gap, so their difficulty cannot be "
            "derived. It predates the measurement; push a labeled bank over it "
            "(deploy/README.md, 'Refreshing the item bank')."
        )
    # Item difficulty is re-derived, because "a pure function of `shallow_gap`"
    # has to be true of the rows and not just of the code that writes new ones —
    # a labeler and a server can be different releases with different curves.
    #
    # Guarded by a read of the thing it changes, so connecting to an up-to-date
    # database takes no write lock: the labeler may be holding one, and opening
    # the server should still just work.
    if conn.execute(
        "SELECT 1 FROM items WHERE rating != difficulty_rating(shallow_gap) LIMIT 1"
    ).fetchone():
        conn.execute(
            "UPDATE items SET rating = difficulty_rating(shallow_gap)"
            " WHERE rating != difficulty_rating(shallow_gap)"
        )
    # When the anchored scale arrived (`rating.RESPONSE_ANCHOR`), because a
    # response's `item_rating_before` on either side of that moment is a
    # snapshot off a different scale and nothing else in the record says where
    # the boundary is. Written only when absent, so it is one-way like the
    # regrade it marks; on a fresh database it truthfully says there was never
    # a pre-anchor row. A pre-anchor backup restored later arrives without the
    # key and gets a new, equally true boundary.
    if not conn.execute("SELECT 1 FROM meta WHERE key = 'anchored_at'").fetchone():
        conn.execute("INSERT INTO meta (key, value) VALUES ('anchored_at', datetime('now'))")
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


def connect_readonly(path: Path = DEFAULT_DB) -> sqlite3.Connection:
    """Open a bank for reading only, refusing one the current curve disowns.

    A report has no business migrating a bank or taking a write lock: it may be
    pointed at a deployment, or run beside a labeler that holds one. But
    `connect` is also where `items.rating` is re-derived, so reading without it
    would quietly report the previous curve's scale — and a retune is exactly
    when someone reads these numbers. Hence the check: the report's bands are
    either current or an error, never stale.

    The regrade itself stays in `connect`, because it is a property of the bank
    and not of whoever happened to look at it.
    """
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
    conn.create_function("difficulty_rating", 1, difficulty_rating, deterministic=True)
    conn.create_function("shallow_gap_of", 1, shallow_gap_of, deterministic=True)
    if conn.execute(
        "SELECT 1 FROM items WHERE rating != difficulty_rating(shallow_gap) LIMIT 1"
    ).fetchone():
        raise RuntimeError(
            f"{path} holds ratings the current difficulty curve does not produce. "
            "Open it once with anything that writes — the server, a labeler, "
            "`push_items` — to regrade it, then re-run this."
        )
    return conn
