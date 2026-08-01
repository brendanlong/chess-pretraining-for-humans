import math
from itertools import pairwise

import pytest

from trainer.label import DEPTH_SHALLOW, MAX_GAP_WP, MIN_GAP_WP
from trainer.rating import (
    CALIB_END_STEP,
    CALIB_START_STEP,
    CALIBRATED_GAP_HI,
    CALIBRATED_GAP_LO,
    GAP_SLOPE,
    RATING_MAX,
    RATING_MIN,
    REFERENCE_DEPTH,
    SELECTION_JITTER,
    TARGET_ACCURACY,
    USER_MAX,
    USER_MIN,
    USER_START,
    calibrate,
    difficulty_rating,
    expected_score,
    regraded_user_rating,
    target_gap,
    target_item_rating,
    update,
)
from trainer.winprob import cp_to_winprob, score_to_winprob


def test_winprob_monotone_and_symmetric():
    assert cp_to_winprob(0) == 0.5
    assert cp_to_winprob(100) > 0.5 > cp_to_winprob(-100)
    assert math.isclose(cp_to_winprob(150), 1 - cp_to_winprob(-150))
    # clamped: a huge eval doesn't push past the +-1000cp asymptote
    assert cp_to_winprob(5000) == cp_to_winprob(1000)


def test_mate_scores_map_to_extremes():
    assert score_to_winprob(None, 3) == cp_to_winprob(1000)
    assert score_to_winprob(None, -2) == cp_to_winprob(-1000)


def test_elo_update_direction():
    assert update(1500, 1500, correct=True) > 1500
    assert update(1500, 1500, correct=False) < 1500


def test_elo_cannot_walk_a_rating_off_the_scale():
    """The staircase clamps, and Elo has to as well: a long run of misses would
    otherwise put a user past the end of the scale, where every target picks the
    same easiest handful of items."""
    r = USER_MIN
    for _ in range(200):
        r = update(r, difficulty_rating(0.5), correct=False)
    assert r == USER_MIN
    r = USER_MAX
    for _ in range(200):
        r = update(r, difficulty_rating(MIN_GAP_WP), correct=True)
    assert r == USER_MAX


def test_difficulty_is_strictly_decreasing_over_every_gap_a_bank_can_hold():
    """The property selection actually needs. Banks mined with a wider
    `--max-gap-wp` exist and reach a 0.65 gap, and any two of those items must
    still be orderable — a clamp at the easy end would make 13.8% of
    the bank indistinguishable and every beginner see the same block."""
    gaps = [g / 1000 for g in range(1, 900)]
    ratings = [difficulty_rating(g) for g in gaps]
    assert all(a > b for a, b in pairwise(ratings))
    # Not just "inside the bounds" — the exponential is positive by
    # construction, so that would be vacuous. The bank's widest real gap has to
    # stay far enough off the floor to leave the easy end room to spread.
    assert difficulty_rating(0.648) > 20
    assert difficulty_rating(0.35) - difficulty_rating(0.648) > 200
    assert difficulty_rating(0.9) > RATING_MIN and difficulty_rating(MIN_GAP_WP) < RATING_MAX


def test_difficulty_is_smooth_where_the_evidence_runs_out():
    """Linear where player-strength data constrains it, decaying past that. The
    join has to be continuous in value *and* slope, or selection would see a
    cliff at an arbitrary gap and items either side would be wrongly spaced."""
    k = CALIBRATED_GAP_HI
    assert difficulty_rating(k - 1e-9) == pytest.approx(difficulty_rating(k + 1e-9))
    below = (difficulty_rating(k) - difficulty_rating(k - 1e-4)) / 1e-4
    above = (difficulty_rating(k + 1e-4) - difficulty_rating(k)) / 1e-4
    assert below == pytest.approx(above, rel=1e-3)
    assert below == pytest.approx(-GAP_SLOPE, rel=1e-3)  # the measured slope


def test_lookahead_is_a_second_axis_and_not_a_tiebreak():
    """Depth has to move an item somewhere selection can tell apart, or it is
    decoration: `pick_item` jitters the target by SELECTION_JITTER, so a step
    smaller than that is inside the noise it is competing with."""
    at = [difficulty_rating(0.2, d) for d in range(1, DEPTH_SHALLOW + 1)]
    assert all(a < b for a, b in pairwise(at))
    assert at[1] - at[0] > SELECTION_JITTER
    # Saturating, not linear: the step the measurement is clearest about is the
    # first one, and each later ply is worth less than the one before it.
    assert all(b - a < prev - before for (before, prev), (a, b) in pairwise(pairwise(at)))
    # And the minority axis. The gap curve is the one with a measurement behind
    # it and the only one the pipeline can steer, so it keeps most of the scale.
    gap_span = difficulty_rating(MIN_GAP_WP, 1) - difficulty_rating(MAX_GAP_WP, 1)
    assert 0 < at[-1] - at[0] < gap_span / 2


def test_an_unmeasured_lookahead_leaves_an_item_where_it_was():
    """Rows labeled before depth was measured keep their difficulty until
    `trainer.backfill_depth` reaches them — being served at the wrong difficulty
    is a smaller wrong than being served as though they were the hardest kind."""
    for gap in (MIN_GAP_WP, 0.2, MAX_GAP_WP):
        assert difficulty_rating(gap, None) == difficulty_rating(gap, REFERENCE_DEPTH)


def test_the_hardest_item_the_labeler_admits_is_not_clamped():
    """RATING_MAX is a guard, not a design feature. If the bank's hard end sat
    on it, every item there would share one difficulty and selection could not
    aim inside them — the failure the curve exists to prevent."""
    hardest = difficulty_rating(MIN_GAP_WP, DEPTH_SHALLOW)
    assert hardest < RATING_MAX
    assert hardest > difficulty_rating(MIN_GAP_WP, DEPTH_SHALLOW - 1)


def test_a_wider_gap_read_deeper_can_be_the_harder_item():
    """What the second axis buys: the top of the scale is reachable at gaps the
    labeler will actually admit, instead of only at ones near its floor."""
    assert difficulty_rating(0.10, 8) > difficulty_rating(MIN_GAP_WP, 1)


def test_difficulty_matches_the_measured_slope_where_it_was_measured():
    """Across the calibrated band the curve is the empirical fit and nothing
    else, so a gap difference converts to rating points at the measured rate."""
    lo, hi = CALIBRATED_GAP_LO, CALIBRATED_GAP_HI
    measured = (difficulty_rating(lo) - difficulty_rating(hi)) / (hi - lo)
    assert measured == pytest.approx(GAP_SLOPE)


def test_regrade_preserves_the_gap_a_user_is_served():
    """The point of the regrade: a rating means nothing except against the
    difficulty it selects, so moving the items has to move the users by the
    same amount or everyone is silently re-aimed."""
    offset = 400 * math.log10(1 / TARGET_ACCURACY - 1)
    for old in (400, 700, 1000, 1400, 1800, 2200, 2600):
        old_gap = (2400 - (old + offset)) / 5000  # the gap the old scale aimed at
        assert target_gap(regraded_user_rating(old)) == pytest.approx(old_gap)


def test_regrade_is_monotone():
    """Two users' ratings can't cross, or the regrade would reorder them."""
    out = [regraded_user_rating(r) for r in range(400, 2600, 25)]
    assert all(a < b for a, b in pairwise(out))


def test_regrade_stays_on_the_scale_for_any_stored_rating():
    """The old scale clamped to [400, 2600], but a hand-edited row or an
    `account` fix could hold anything, and a rating outside the bounds would
    make `calibrate` snap on the first answer."""
    for old in (-1e9, 0, 400, 2600, 1e9, float("inf")):
        assert USER_MIN <= regraded_user_rating(old) <= USER_MAX


def test_every_user_on_the_scale_is_aimed_at_an_item_the_bank_can_hold():
    """A target below the easiest item hands every beginner the same clamped
    block, so the whole user range has to map to a real gap."""
    for r in range(USER_MIN, USER_MAX, 50):
        gap = target_gap(r)
        assert 0 <= gap <= MAX_GAP_WP, f"rating {r} aims at gap {gap}"


def test_target_rating_hits_target_accuracy():
    # strip the jitter by averaging
    targets = [target_item_rating(1500) for _ in range(2000)]
    mean_target = sum(targets) / len(targets)
    assert math.isclose(expected_score(1500, mean_target), TARGET_ACCURACY, abs_tol=0.02)


def test_expected_score_bounds():
    assert expected_score(2000, 1000) > 0.95
    assert expected_score(1000, 2000) < 0.05


def test_calibration_climbs_fast_for_strong_players():
    r, step = USER_START, CALIB_START_STEP
    trials = 0
    while r < 2200:
        r, step = calibrate(r, step, correct=True)
        trials += 1
    assert trials <= 10  # an expert reaches expert territory within ~10 trials


def test_calibration_settles_fast_for_beginners():
    r, step = USER_START, CALIB_START_STEP
    misses = 0
    while step >= CALIB_END_STEP:
        r, step = calibrate(r, step, correct=False)
        misses += 1
    assert misses <= 3  # a few misses and calibration hands off to Elo
    assert USER_MIN <= r <= USER_START


def test_calibration_respects_bounds():
    r, _ = calibrate(USER_MAX, CALIB_START_STEP, correct=True)
    assert r == USER_MAX
    r, _ = calibrate(USER_MIN, CALIB_START_STEP, correct=False)
    assert r == USER_MIN
