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
from trainer.rating import difficulty_rating


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
    # An off-formula rating, purely so a row that merge must not rewrite is
    # distinguishable from an incoming one.
    conn.execute("UPDATE items SET rating = 1800 WHERE id = 1")
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


def test_export_refuses_to_overwrite(tmp_path):
    out = tmp_path / "export.db"
    out.write_bytes(b"")
    with pytest.raises(SystemExit):
        export(live_db(tmp_path), out)


def test_merge_adds_new_positions_and_leaves_the_record_alone(tmp_path):
    live = live_db(tmp_path)
    before = tuple(connect(live).execute("SELECT * FROM items WHERE id = 1").fetchone())
    incoming = bank(tmp_path / "fresh.db", ["8", "7P", "6P1"])  # two overlap, one is new

    assert merge(live, incoming) == (1, 2, 0)

    conn = connect(live)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
    # Ids the responses point at are untouched, and so is the row itself:
    # relabelling an item under the answers already given to it would make
    # those answers uninterpretable.
    assert tuple(conn.execute("SELECT * FROM items WHERE id = 1").fetchone()) == before
    assert conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1


def unmeasured(path):
    """A live bank as it stands before any depth has reached it."""
    conn = connect(path)
    conn.execute("UPDATE items SET solution_depth = NULL, rating = difficulty_rating(gap_wp, NULL)")
    conn.commit()
    conn.close()
    return path


def test_merge_fills_in_a_lookahead_depth_the_live_bank_never_measured(tmp_path):
    """The one thing a held position takes from the incoming bank.

    Depth needs Stockfish, which only the pipeline machine has, and the live
    bank can't be replaced wholesale — so without this the rows that predate
    the measurement would be served at gap-only difficulty forever.
    """
    live = unmeasured(live_db(tmp_path))
    incoming = bank(tmp_path / "fresh.db", ["8", "7P"])

    assert merge(live, incoming) == (0, 2, 2)
    # Read on a raw connection: `connect` re-derives `rating`, so opening the
    # file the ordinary way would repair exactly what this is checking. The live
    # server holds its connections open past a merge and re-derives on neither,
    # so a difficulty this doesn't write is a difficulty nobody sees.
    raw = sqlite3.connect(live)
    raw.row_factory = sqlite3.Row
    rows = raw.execute("SELECT solution_depth, rating FROM items ORDER BY id").fetchall()
    assert [r["solution_depth"] for r in rows] == [2, 2]
    assert rows[0]["rating"] == difficulty_rating(0.10, 2) != difficulty_rating(0.10)


def test_merge_retires_a_position_no_search_gets_right(tmp_path):
    """Unlearnable is a depth verdict too, and it has to travel as one — even
    onto a live row an older, hash-warm filter had passed as learnable."""
    live = unmeasured(live_db(tmp_path))
    incoming = bank(tmp_path / "fresh.db", [])
    conn = connect(incoming)
    add_item(conn, FEN_TMPL.format("8"), solution_depth=0, learnable=0)
    conn.commit()
    conn.close()

    assert merge(live, incoming) == (0, 1, 1)
    served = connect(live).execute("SELECT learnable FROM items ORDER BY id").fetchall()
    assert [r["learnable"] for r in served] == [0, 1]


def test_merge_re_measures_a_legacy_row_the_old_filter_had_already_rejected(tmp_path):
    """`learnable = 0` on a live row is the *old* single-depth check's verdict,
    taken on a hash the deep pass had just filled. It carries no depth, so the
    ladder has never really run on it, and both fill-in paths have to agree that
    it still needs to."""
    live = unmeasured(live_db(tmp_path))
    conn = connect(live)
    conn.execute("UPDATE items SET learnable = 0 WHERE id = 1")
    conn.commit()
    conn.close()
    incoming = bank(tmp_path / "fresh.db", ["8"])  # the same position, now measured

    assert merge(live, incoming) == (0, 1, 1)
    row = connect(live).execute("SELECT solution_depth, learnable FROM items WHERE id = 1")
    assert tuple(row.fetchone()) == (2, 1)  # back in the bank, on a real measurement


def test_merge_leaves_a_depth_already_measured_alone(tmp_path):
    """Re-measuring one is relabelling, which is what this tool refuses. That
    includes the 0 that says no depth settles it: it is an answer, not a gap."""
    live = live_db(tmp_path)
    conn = connect(live)
    conn.execute("UPDATE items SET solution_depth = 0, learnable = 0 WHERE id = 1")
    conn.commit()
    conn.close()
    incoming = bank(tmp_path / "fresh.db", ["8", "7P"])

    assert merge(live, incoming) == (0, 2, 0)
    rows = connect(live).execute("SELECT solution_depth FROM items ORDER BY id").fetchall()
    assert [r["solution_depth"] for r in rows] == [0, 2]


def test_merge_carries_its_items_across_a_source_that_predates_the_column(tmp_path):
    """Schema skew is this tool's premise, so the fill-in has to stand down
    rather than raise — a failure here rolls the inserts back with it, and the
    operator's push silently delivers nothing."""
    live = unmeasured(live_db(tmp_path))
    incoming = bank(tmp_path / "fresh.db", ["6P1"])
    conn = connect(incoming)
    conn.execute("ALTER TABLE items DROP COLUMN solution_depth")
    conn.commit()
    conn.close()

    assert merge(live, incoming) == (1, 0, 0)
    conn = connect(live)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 3
    assert conn.execute("SELECT solution_depth FROM items WHERE id = 1").fetchone()[0] is None


def test_merge_dry_run_writes_nothing(tmp_path):
    """Including the fill-in, which is the half that would otherwise land on a
    run whose whole point is to be asked about first."""
    live = unmeasured(live_db(tmp_path))
    incoming = bank(tmp_path / "fresh.db", ["8", "6P1"])

    assert merge(live, incoming, dry_run=True) == (1, 1, 1)
    conn = connect(live)
    assert conn.execute("SELECT COUNT(*) FROM items").fetchone()[0] == 2
    assert (
        conn.execute("SELECT COUNT(*) FROM items WHERE solution_depth IS NULL").fetchone()[0] == 2
    )


def test_merge_survives_a_source_missing_a_column(tmp_path):
    """The two databases are upgraded at different times, by construction."""
    live = live_db(tmp_path)
    incoming = bank(tmp_path / "fresh.db", ["6P1"])
    conn = connect(incoming)
    conn.execute("ALTER TABLE items DROP COLUMN mover_elo")
    conn.commit()
    conn.close()

    assert merge(live, incoming) == (1, 0, 0)
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
    # Including the WAL sidecars: a retry would otherwise open a fresh database
    # on top of a stale one's journal.
    assert [p.name for p in tmp_path.glob("export.db*")] == []


def test_roundtrip(tmp_path):
    """What the refresh actually runs: export here, merge there."""
    out = tmp_path / "export.db"
    export(bank(tmp_path / "fresh.db", ["8", "7P", "6P1"]), out)
    assert merge(live_db(tmp_path), out) == (1, 2, 0)
