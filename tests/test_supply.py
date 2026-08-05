"""Reporting what the bank can serve.

The report exists because selection can't fail, so a hole in the bank is
invisible from inside the app. What these tests hold is that it sees a hole the
app would have hidden, and that it says the two things the difficulty number
can't: how much of a band is made of items whose surface actively misleads, and
where the deep gap — the only thing mining can steer — actually lands.
"""

import sqlite3

import pytest

from tests.conftest import ITEM, add_item
from trainer.db import connect, connect_readonly
from trainer.rating import _TARGET_OFFSET, _gap_for_difficulty, difficulty_rating
from trainer.supply import SELECTION_POOL, band, band_misleading, by_gap, pool_drift


def bank(tmp_path, shallow_gaps, learnable=1, **overrides):
    """A bank built from shallow gaps, which is what difficulty is a function of.

    The ladder is written to match, because `db.connect` derives `shallow_gap`
    from it wherever the column is empty — a fixture whose two disagreed would
    be testing a row the migration is about to rewrite.
    """
    conn = connect(tmp_path / "supply.db")
    for i, gap in enumerate(shallow_gaps):
        # The fen is never parsed here and unique is all the table asks.
        add_item(
            conn,
            f"position-{i}",
            shallow_gap=gap,
            gap_ladder=" ".join([f"{gap:.4f}"] * 8),
            rating=difficulty_rating(gap),
            learnable=learnable,
            **overrides,
        )
    conn.commit()
    return conn


def test_a_band_counts_only_what_selection_would_reach_for(tmp_path):
    target = 1400 + _TARGET_OFFSET
    here, elsewhere = _gap_for_difficulty(target), _gap_for_difficulty(target) + 0.2
    conn = bank(tmp_path, [here] * 5 + [elsewhere] * 40)
    assert band(conn, target) == 5


def test_an_unlearnable_item_is_not_supply(tmp_path):
    """It is in the bank and never served, so counting it would overstate."""
    target = 1400 + _TARGET_OFFSET
    conn = bank(tmp_path, [_gap_for_difficulty(target)] * 5, learnable=0)
    assert band(conn, target) == 0


def test_a_band_reports_how_much_of_it_is_actively_misleading(tmp_path):
    """One band, two kinds of item: gaps too narrow to see, and gaps pointing
    the wrong way. They sit within a jitter of each other because the sign of
    the gap is where the scale crosses, and only the second kind trains the
    reflex to distrust what a position looks like — which the count can't say
    and the difficulty number can't either."""
    conn = bank(tmp_path, [-0.010, -0.005, 0.005, 0.010])
    target = difficulty_rating(0.0)

    assert band(conn, target) == 4  # all four are one band
    assert band_misleading(conn, target) == 0.5


def test_a_band_of_nothing_reports_no_share_rather_than_zero(tmp_path):
    """Zero would read as "none of this band misleads", which is a claim about
    a band that isn't there."""
    conn = bank(tmp_path, [0.30])
    assert band_misleading(conn, difficulty_rating(0.0)) != band_misleading(
        conn, difficulty_rating(0.30)
    )
    assert str(band_misleading(conn, difficulty_rating(0.0))) == "nan"


def test_drift_is_what_a_thin_band_costs(tmp_path):
    """A band with nothing in it doesn't fail — it serves the wrong difficulty."""
    target = 1400 + _TARGET_OFFSET
    conn = bank(tmp_path, [_gap_for_difficulty(target) + 0.2] * SELECTION_POOL)
    assert band(conn, target) == 0
    assert pool_drift(conn, target) > 500


def test_the_gap_table_says_where_a_deep_gap_bin_landed(tmp_path):
    """Mining steers the deep gap and difficulty is a function of the shallow
    one, so a bin no longer maps to a band. Reporting the quartiles it produced
    is the honest replacement for an order stated in items."""
    conn = connect(tmp_path / "supply.db")
    # One bin, two very different shallow gaps: the same mining order, scattered.
    for i, shallow in enumerate([0.02] * 5 + [0.30] * 5):
        add_item(
            conn,
            f"position-{i}",
            gap_wp=0.22,
            shallow_gap=shallow,
            gap_ladder=" ".join([f"{shallow:.4f}"] * 8),
            rating=difficulty_rating(shallow),
        )
    conn.commit()

    row = next(r for r in by_gap(conn, 0.05) if r["lo"] <= 0.22 < r["hi"])
    assert row["items"] == 10
    # Sorted ascending, so p25 is the *easier* quartile — one mining order,
    # landing more than a band apart at each end.
    assert row["p75"] - row["p25"] > 500, "the bin should visibly scatter"
    assert row["p25"] == difficulty_rating(0.30) and row["p75"] == difficulty_rating(0.02)


def test_a_gap_range_nobody_has_mined_is_still_a_row(tmp_path):
    """The emptiest band there is, and bounding the table by the data hides it."""
    conn = bank(tmp_path, [0.05] * 10, gap_wp=0.05)
    rows = by_gap(conn, 0.05)

    assert max(row["hi"] for row in rows) > 0.6
    assert any(row["items"] == 0 for row in rows)


def test_the_report_cannot_write_to_the_bank_it_reads(tmp_path):
    """The point of opening read-only: this can be aimed at a live bank, beside
    a server or a labeler holding the write lock."""
    bank(tmp_path, [0.20] * 3).close()
    conn = connect_readonly(tmp_path / "supply.db")

    assert band(conn, difficulty_rating(0.20)) == 3
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("UPDATE items SET rating = 0")


def test_a_bank_on_the_previous_curve_is_refused_not_reported(tmp_path):
    """`connect` regrades on open and this cannot, so the stale scale would be
    invisible — and a retune is exactly when someone reads these bands."""
    conn = bank(tmp_path, [0.20] * 3)
    conn.execute("UPDATE items SET rating = rating + 131")  # what an anchor shift looks like
    conn.commit()
    conn.close()

    with pytest.raises(RuntimeError, match="does not produce"):
        connect_readonly(tmp_path / "supply.db")

    # And the regrade that fixes it is still `connect`'s job, not the report's.
    connect(tmp_path / "supply.db").close()
    assert band(connect_readonly(tmp_path / "supply.db"), difficulty_rating(0.20)) == 3


def test_the_fixture_is_the_shape_the_report_reads(tmp_path):
    """`add_item`'s defaults are one difficulty; these tests need a spread."""
    conn = bank(tmp_path, [0.02, 0.30])
    stored = sorted(row["shallow_gap"] for row in conn.execute("SELECT shallow_gap FROM items"))
    assert stored == [0.02, 0.30] != [ITEM["shallow_gap"]] * 2
