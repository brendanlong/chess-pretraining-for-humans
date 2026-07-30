import math
from itertools import pairwise

import pytest

from trainer.rating import (
    CALIB_END_STEP,
    CALIB_START_STEP,
    CALIBRATED_GAP_HI,
    CALIBRATED_GAP_LO,
    GAP_SLOPE,
    RATING_MAX,
    RATING_MIN,
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


def test_difficulty_is_strictly_decreasing_over_every_gap_a_bank_can_hold():
    """The property selection actually needs. Banks mined with a wider
    `--max-gap-wp` exist and reach a 0.65 gap, and any two of those items must
    still be orderable — a clamp flattening the easy end is what made 13.8% of
    the bank indistinguishable and every beginner see the same block."""
    gaps = [g / 1000 for g in range(1, 900)]
    ratings = [difficulty_rating(g) for g in gaps]
    assert all(a > b for a, b in pairwise(ratings))
    assert min(ratings) > RATING_MIN and max(ratings) < RATING_MAX  # nothing clamps


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
