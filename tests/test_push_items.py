"""Refreshing a deployment's item bank.

The bank is disposable, `responses` is not, and both live in one file — so the
thing worth testing is that pushing a newly labeled bank at a live database
adds positions without disturbing anything earned there.
"""

import sqlite3

import pytest

from tests.conftest import FEN_TMPL, add_item
from trainer import push_items
from trainer.db import connect
from trainer.push_items import export, merge


def bank(path, ranks):
    conn = connect(path)
    for rank in ranks:
        add_item(conn, FEN_TMPL.format(rank))
    conn.commit()
    conn.close()
    return path


def loose_bank(path):
    """A bank whose items table predates the constraints the live one has."""
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE items (fen TEXT, best_uci TEXT, learnable INTEGER)")
    conn.execute("INSERT INTO items VALUES ('8/8/8/8/8/8/8/K6k w - - 0 1', 'a1a2', NULL)")
    conn.commit()
    conn.close()
    return path


def live_db(tmp_path):
    """A deployment: two items, one of them answered."""
    path = bank(tmp_path / "live.db", ["8", "7P"])
    conn = connect(path)
    conn.execute("INSERT INTO users (name, rating) VALUES ('brendan', 1200)")
    conn.execute(
        "INSERT INTO responses (user_id, item_id, choice_uci, correct) VALUES (1, 1, 'e2e4', 1)"
    )
    conn.execute("UPDATE items SET rating = 1800, attempts = 1, correct = 1 WHERE id = 1")
    conn.commit()
    conn.close()
    return path


def test_export_carries_items_and_nothing_else(tmp_path):
    out = tmp_path / "export.db"
    assert export(live_db(tmp_path), out) == 2
    conn = connect(out)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 0
    # The tally of answers given on the source deployment doesn't travel: the
    # answers themselves aren't in the export.
    assert tuple(conn.execute("SELECT attempts, correct FROM items").fetchone()) == (0, 0)


def test_export_refuses_to_overwrite(tmp_path):
    out = tmp_path / "export.db"
    out.write_bytes(b"")
    with pytest.raises(SystemExit):
        export(live_db(tmp_path), out)


def test_merge_adds_new_positions_and_leaves_the_record_alone(tmp_path):
    live = live_db(tmp_path)
    incoming = bank(tmp_path / "fresh.db", ["8", "7P", "6P1"])  # two overlap, one is new

    added, skipped = merge(live, incoming)
    assert (added, skipped) == (1, 2)

    conn = connect(live)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
    # Ids the responses point at are untouched, and so is the difficulty the
    # live deployment's answers earned for that item.
    item = conn.execute("SELECT * FROM items WHERE id = 1").fetchone()
    assert (item["rating"], item["attempts"], item["correct"]) == (1800, 1, 1)
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def test_merge_dry_run_writes_nothing(tmp_path):
    live = live_db(tmp_path)
    incoming = bank(tmp_path / "fresh.db", ["6P1"])

    assert merge(live, incoming, dry_run=True) == (1, 0)
    assert connect(live).execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_merge_survives_a_source_missing_a_column(tmp_path):
    """The two databases are upgraded at different times, by construction."""
    live = live_db(tmp_path)
    incoming = bank(tmp_path / "fresh.db", ["6P1"])
    conn = connect(incoming)
    conn.execute("ALTER TABLE items DROP COLUMN mover_elo")
    conn.commit()
    conn.close()

    assert merge(live, incoming) == (1, 0)
    added = connect(live).execute("SELECT * FROM items WHERE id = 3").fetchone()
    assert added["mover_elo"] is None
    assert added["fen"] == FEN_TMPL.format("6P1")


def test_a_bad_source_reports_itself_and_changes_nothing(tmp_path):
    """The failure has to survive the cleanup that follows it.

    SQLite won't detach inside a transaction, and the failed insert leaves one
    open — so a careless `finally` raises a lock error over the top of the real
    one, and the operator is left debugging the wrong thing.
    """
    live = live_db(tmp_path)
    # The loose bank has none of the columns the live one requires.
    with pytest.raises(sqlite3.IntegrityError):
        merge(live, loose_bank(tmp_path / "bad.db"))

    assert connect(live).execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2


def test_a_failed_export_leaves_no_file_to_trip_over(tmp_path, monkeypatch):
    """Refusing to overwrite is only safe if a failure doesn't leave a file.

    The realistic failure is the disk (the destination is created before it's
    filled), so inject one rather than contrive a bad bank.
    """
    live = live_db(tmp_path)
    out = tmp_path / "export.db"

    def boom(*args):
        raise OSError("no space left on device")

    monkeypatch.setattr(push_items, "shared_columns", boom)
    with pytest.raises(OSError):
        push_items.export(live, out)
    assert not out.exists()


def test_roundtrip(tmp_path):
    """What the refresh actually runs: export here, merge there."""
    out = tmp_path / "export.db"
    export(bank(tmp_path / "fresh.db", ["8", "7P", "6P1"]), out)
    assert merge(live_db(tmp_path), out) == (1, 2)
