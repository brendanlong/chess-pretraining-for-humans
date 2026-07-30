"""Reporting what the bank can serve.

The report exists because selection can't fail: `pick_item` always returns the
30 nearest items, so a hole in the bank is invisible from inside the app. What
these tests hold is that the report sees a hole the app would have hidden.
"""

from tests.conftest import ITEM
from trainer.db import connect
from trainer.rating import _TARGET_OFFSET, difficulty_rating
from trainer.supply import SELECTION_POOL, band, band_width_in_gap, by_gap, pool_drift


def bank(tmp_path, gaps, learnable=1):
    conn = connect(tmp_path / "supply.db")
    for i, gap in enumerate(gaps):
        conn.execute(
            """INSERT INTO items (fen, best_uci, distractor_uci, distractor_source,
                 cp_best, cp_distractor, wp_best, wp_distractor, gap_wp, learnable,
                 depth_deep, depth_shallow, rating, ply, game_url, mover_elo, time_control)
               VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
                 :cp_best, :cp_distractor, :wp_best, :wp_distractor, :gap_wp, :learnable,
                 :depth_deep, :depth_shallow, :rating, :ply, :game_url, :mover_elo,
                 :time_control)""",
            {
                **ITEM,
                # Never parsed by the report, and unique is all the table asks.
                "fen": f"position-{i}",
                "gap_wp": gap,
                "rating": difficulty_rating(gap),
                "learnable": learnable,
            },
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


def test_the_easy_end_needs_fewer_items_for_the_same_band(tmp_path):
    """Past the curve's knee a band spans more gap, so a bin there goes further."""
    assert band_width_in_gap(0.45) > 3 * band_width_in_gap(0.25)
