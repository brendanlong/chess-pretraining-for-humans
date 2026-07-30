"""Reporting what the bank can serve.

The report exists because selection can't fail, so a hole in the bank is
invisible from inside the app. What these tests hold is that it sees a hole the
app would have hidden — including the two the report itself can hide: a band
nobody has mined for, and the easy end, where the curve flattens far enough
that a careless width asks for a tenth of what the floor needs.
"""

from tests.conftest import ITEM, add_item
from trainer.db import connect
from trainer.rating import _TARGET_OFFSET, difficulty_rating, target_gap
from trainer.supply import SELECTION_POOL, band, band_width_in_gap, by_gap, pool_drift


def bank(tmp_path, gaps, learnable=1):
    conn = connect(tmp_path / "supply.db")
    for i, gap in enumerate(gaps):
        # The fen is never parsed here and unique is all the table asks.
        add_item(
            conn,
            f"position-{i}",
            gap_wp=gap,
            rating=difficulty_rating(gap),
            learnable=learnable,
        )
    conn.commit()
    return conn


def test_a_band_counts_only_what_selection_would_reach_for(tmp_path):
    # Two clumps a long way apart on the scale: one at the gap a 1400 is aimed
    # at, one far easier.
    here, elsewhere = 0.238, 0.45
    conn = bank(tmp_path, [here] * 5 + [elsewhere] * 40)
    assert band(conn, 1400 + _TARGET_OFFSET) == 5


def test_an_unlearnable_item_is_not_supply(tmp_path):
    """It is in the bank and never served, so counting it would overstate."""
    conn = bank(tmp_path, [0.238] * 5, learnable=0)
    assert band(conn, 1400 + _TARGET_OFFSET) == 0


def test_drift_is_what_a_thin_band_costs(tmp_path):
    """A band with nothing in it doesn't fail — it serves the wrong difficulty."""
    target = 1400 + _TARGET_OFFSET
    conn = bank(tmp_path, [0.45] * SELECTION_POOL)
    assert band(conn, target) == 0
    assert pool_drift(conn, target) > 500


def test_the_gap_table_prices_the_hole_in_mining_units(tmp_path):
    """The shortfall is what a refill run is sized from, so it has to be real."""
    step, floor = 0.05, 100
    full = round(floor * step / band_width_in_gap(0.225))
    conn = bank(tmp_path, [0.225] * full + [0.275] * (full // 2))
    rows = {round(r["lo"], 2): r for r in by_gap(conn, step, floor)}

    assert rows[0.20]["short"] < 1
    assert round(rows[0.25]["short"]) == full - full // 2


def test_labeling_what_the_gap_table_asks_for_reaches_the_floor(tmp_path):
    """The two tables are one claim; this is the claim.

    It is the easy end that breaks it. Past the knee the curve flattens without
    limit, so a band centred out there reaches gaps no position has — and a
    width that believes it asks for a fraction of what the floor really needs,
    while the middle of the scale stays right and hides that it doesn't.
    """
    step, floor = 0.005, 200
    empty = connect(tmp_path / "empty.db")
    plan = by_gap(empty, step, floor)

    # Spread each bin's order across the bin rather than stacking it on the
    # centre: a band is only a few bins wide, so lumped items make the count
    # depend on where the band's edges fall between lumps.
    gaps = [
        row["lo"] + (i + 0.5) * step / round(row["need"])
        for row in plan
        for i in range(round(row["need"]))
    ]
    conn = bank(tmp_path, gaps)

    for user in (300, 900, 1400, 2000, 2500):
        assert band(conn, user + _TARGET_OFFSET) >= 0.9 * floor, (
            f"user {user} (gap {target_gap(user):.3f}) is short after filling the order"
        )


def test_a_gap_range_nobody_has_mined_is_still_a_row(tmp_path):
    """The emptiest band there is, and bounding the table by the data hides it."""
    conn = bank(tmp_path, [0.05] * 10)
    rows = by_gap(conn, 0.05, 100)

    assert max(row["hi"] for row in rows) > target_gap(300)
    assert any(row["items"] == 0 and row["short"] > 0 for row in rows)


def test_the_easy_end_needs_fewer_items_for_the_same_band(tmp_path):
    """Past the curve's knee a band spans more gap, so a bin there goes further.

    Sampled past the point where the unclipped width diverges, since that is
    where the difference stops being a ratio and starts being nonsense.
    """
    assert 3 * band_width_in_gap(0.25) < band_width_in_gap(0.55) < 1.0


def test_the_fixture_is_the_shape_the_report_reads(tmp_path):
    """`add_item`'s defaults are one difficulty; these tests need a spread."""
    conn = bank(tmp_path, [0.1, 0.4])
    stored = sorted(row["gap_wp"] for row in conn.execute("SELECT gap_wp FROM items"))
    assert stored == [0.1, 0.4] != [ITEM["gap_wp"]] * 2
