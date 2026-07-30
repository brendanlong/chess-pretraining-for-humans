"""Upgrading a database that predates accounts.

`responses` is the experimental record and is not rebuildable, so the path
that touches it deserves a test of its own — the rest of the suite always
starts from a fresh schema.
"""

import sqlite3

from trainer import account, auth
from trainer.db import connect

from .conftest import FEN_RANKS, FEN_TMPL, add_item

# The `users` table as it stood before this branch, plus the one column the
# earlier calib_step migration added.
PRE_ACCOUNTS_SCHEMA = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    rating REAL NOT NULL DEFAULT 700,
    calib_step REAL NOT NULL DEFAULT 250,
    attempts INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE responses (
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
INSERT INTO users (name, rating, calib_step, attempts) VALUES ('brendan', 1420.0, 20.0, 3);
INSERT INTO responses (user_id, item_id, choice_uci, correct) VALUES (1, 1, 'e2e4', 1);
INSERT INTO responses (user_id, item_id, choice_uci, correct) VALUES (1, 2, 'a2a3', 0);
"""


def old_db(tmp_path):
    path = tmp_path / "legacy.db"
    legacy = sqlite3.connect(path)
    legacy.executescript(PRE_ACCOUNTS_SCHEMA)
    legacy.commit()
    legacy.close()
    return path


def test_migration_preserves_users_and_responses(tmp_path):
    path = old_db(tmp_path)
    conn = connect(path)

    user = conn.execute("SELECT * FROM users WHERE name = 'brendan'").fetchone()
    assert user["rating"] == 1420.0
    assert user["attempts"] == 3
    assert user["password_hash"] is None  # a legacy row is just a guest
    assert user["created_at"] is not None  # backfilled, not left NULL
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


def test_migration_drops_the_columns_that_carried_answers_into_difficulty(tmp_path):
    """Difficulty is a function of the item now, so the counters answers used to
    feed and the post-answer item rating go — without disturbing the responses
    sitting beside them, which are the experimental record."""
    path = old_db(tmp_path)
    conn = connect(path)  # brings the schema up to date, then we age it back
    add_item(conn, FEN_TMPL.format(FEN_RANKS[0]))
    conn.execute("ALTER TABLE items ADD COLUMN attempts INTEGER NOT NULL DEFAULT 7")
    conn.execute("ALTER TABLE items ADD COLUMN correct INTEGER NOT NULL DEFAULT 5")
    conn.commit()
    conn.close()

    conn = connect(path)
    item_cols = {row[1] for row in conn.execute("PRAGMA table_info(items)")}
    assert {"attempts", "correct"}.isdisjoint(item_cols)
    assert "rating" in item_cols  # the difficulty itself stays
    response_cols = {row[1] for row in conn.execute("PRAGMA table_info(responses)")}
    assert "item_rating_after" not in response_cols
    assert "item_rating_before" in response_cols
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 1


def test_migration_is_idempotent_and_takes_no_write_lock_when_current(tmp_path):
    """Connecting to an up-to-date database must not write: the labeler can be
    holding the write lock, and opening the server should still just work."""
    path = old_db(tmp_path)
    connect(path).close()

    blocker = sqlite3.connect(path, isolation_level=None)
    blocker.execute("BEGIN IMMEDIATE")  # hold the write lock, as the labeler does
    try:
        connect(path).close()
    finally:
        blocker.execute("ROLLBACK")
        blocker.close()


def test_legacy_rows_cannot_be_logged_into_by_guessing_the_name(tmp_path):
    conn = connect(old_db(tmp_path))
    user = auth.find_by_username(conn, "brendan")
    assert user is not None
    assert auth.credential_for(user) is None  # no password to check against
    assert auth.verify_password(auth.credential_for(user), "brendan") is False


def test_migration_is_idempotent(tmp_path):
    path = old_db(tmp_path)
    connect(path).close()
    conn = connect(path)  # second open must not fail or duplicate anything
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_sessions_table_is_rebuilt_with_cascade(tmp_path):
    """ON DELETE CASCADE arrived after databases existed, and SQLite can't add
    a constraint in place — so connect() rebuilds the table, keeping resolvable
    sessions and shedding orphans left from before the key was enforced."""
    path = tmp_path / "precascade.db"
    raw = sqlite3.connect(path)
    raw.executescript(
        PRE_ACCOUNTS_SCHEMA
        + """
        CREATE TABLE sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(id),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            last_seen TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT INTO sessions (token_hash, user_id) VALUES ('kept', 1);
        INSERT INTO sessions (token_hash, user_id) VALUES ('orphan', 999);
        """
    )
    raw.commit()
    raw.close()

    conn = connect(path)

    fk = conn.execute("PRAGMA foreign_key_list(sessions)").fetchone()
    assert fk["on_delete"] == "CASCADE"
    assert [r[0] for r in conn.execute("SELECT token_hash FROM sessions")] == ["kept"]
    connect(path).close()  # and the rebuild doesn't run twice


def test_sweep_never_touches_a_legacy_row_with_responses(tmp_path):
    conn = connect(old_db(tmp_path))
    conn.execute("UPDATE users SET created_at = datetime('now', '-99 days')")
    conn.commit()
    auth.sweep(conn)
    assert auth.find_by_username(conn, "brendan") is not None


def test_set_password_cli_adopts_a_legacy_row(tmp_path, monkeypatch, capsys):
    path = old_db(tmp_path)
    monkeypatch.setattr(account.getpass, "getpass", lambda _: "legacypass1")

    assert account.main(["--db", str(path), "set-password", "brendan"]) == 0

    conn = connect(path)
    user = auth.find_by_username(conn, "brendan")
    assert user is not None
    assert auth.verify_password(auth.credential_for(user), "legacypass1")
    assert user["attempts"] == 3  # history intact
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


def test_set_password_cli_rejects_a_mismatch(tmp_path, monkeypatch):
    path = old_db(tmp_path)
    answers = iter(["one-password", "another-password"])
    monkeypatch.setattr(account.getpass, "getpass", lambda _: next(answers))

    assert account.main(["--db", str(path), "set-password", "brendan"]) == 1
    assert connect(path).execute("SELECT password_hash FROM users").fetchone()[0] is None


def test_set_password_cli_refuses_an_unknown_user(tmp_path):
    assert account.main(["--db", str(old_db(tmp_path)), "set-password", "nobody"]) == 1


def test_set_password_cli_will_not_take_an_existing_name(tmp_path, monkeypatch):
    path = old_db(tmp_path)
    conn = connect(path)
    conn.execute("INSERT INTO users (name, created_at) VALUES ('taken', datetime('now'))")
    conn.commit()
    monkeypatch.setattr(account.getpass, "getpass", lambda _: "legacypass1")

    args = ["--db", str(path), "set-password", "brendan", "--rename-to", "TAKEN"]
    assert account.main(args) == 1


def test_delete_cli_erases_a_legacy_row_and_its_responses(tmp_path, monkeypatch, capsys):
    """The operator path exists for the rows the app can't reach — no password
    to re-enter, so no in-app deletion."""
    path = old_db(tmp_path)
    conn = connect(path)
    # A bystander with responses of their own, so a table wipe can't pass here.
    # (The bystander's response needs a real item — foreign keys are enforced —
    # unlike the legacy rows, which predate enforcement and are tolerated.)
    add_item(conn, FEN_TMPL.format(FEN_RANKS[0]))
    conn.execute("INSERT INTO users (name, created_at) VALUES ('keeper', datetime('now'))")
    conn.execute(
        """INSERT INTO responses (user_id, item_id, choice_uci, correct)
           SELECT id, 1, 'e2e4', 1 FROM users WHERE name = 'keeper'"""
    )
    conn.commit()
    monkeypatch.setattr("builtins.input", lambda _: "brendan")

    assert account.main(["--db", str(path), "delete", "brendan"]) == 0

    conn = connect(path)
    assert auth.find_by_username(conn, "brendan") is None
    assert auth.find_by_username(conn, "keeper") is not None
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
    assert "2 responses" in capsys.readouterr().out


def test_delete_cli_needs_the_name_typed_back(tmp_path, monkeypatch):
    path = old_db(tmp_path)
    monkeypatch.setattr("builtins.input", lambda _: "brendan-old")

    assert account.main(["--db", str(path), "delete", "brendan"]) == 1

    conn = connect(path)
    assert auth.find_by_username(conn, "brendan") is not None
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 2


def test_delete_cli_refuses_an_unknown_user(tmp_path):
    assert account.main(["--db", str(old_db(tmp_path)), "delete", "nobody", "--yes"]) == 1


def test_list_cli_reports_accounts_and_guests(tmp_path, capsys):
    assert account.main(["--db", str(old_db(tmp_path)), "list"]) == 0
    out = capsys.readouterr().out
    assert "brendan" in out and "3 trials" in out and "guest" in out
