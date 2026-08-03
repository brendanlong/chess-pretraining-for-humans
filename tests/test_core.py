import math
from itertools import pairwise

import pytest

from trainer.rating import (
    _TARGET_OFFSET,
    CALIB_END_STEP,
    CALIB_START_STEP,
    CALIBRATED_GAP_HI,
    CALIBRATED_GAP_LO,
    GAP_SLOPE,
    HARD_CEILING,
    KNEE_DIFFICULTY,
    RATING_MAX,
    RESPONSE_ANCHOR,
    SHALLOW_PLIES,
    TARGET_ACCURACY,
    USER_MAX,
    USER_MIN,
    USER_START,
    _gap_for_difficulty,
    calibrate,
    difficulty_rating,
    expected_score,
    shallow_gap_of,
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
        r = update(r, difficulty_rating(-0.3), correct=True)
    assert r == USER_MAX


def test_a_misleading_item_is_harder_than_a_merely_invisible_one():
    """The whole reason difficulty reads the shallow end of the search. A gap
    that is negative there is a position whose surface recommends the wrong
    move, and it has to rate above one that is merely too small to see."""
    assert difficulty_rating(-0.10) > difficulty_rating(0.0) > difficulty_rating(0.10)


def test_difficulty_is_strictly_decreasing_over_every_shallow_gap_there_is():
    """Including well past both ends of the evidence. A range of gaps that share
    one rating is a range selection cannot aim inside, and it lands on whoever
    sits there."""
    gaps = [g / 1000 for g in range(-500, 900)]
    assert all(a > b for a, b in pairwise([difficulty_rating(g) for g in gaps]))


def test_the_range_users_are_aimed_into_is_spread_rather_than_saturated():
    """Both tails are asymptotic by construction, so "inside the bounds" would be
    vacuous — and past the evidence they compress hard, which is the price of
    staying ordered out there. What has to hold is about the part selection can
    actually aim at: across the whole user scale, one jitter of difficulty is
    still a real difference in gap, or two neighbouring bands are the same band.
    """
    assert HARD_CEILING <= RATING_MAX  # the guard never binds
    targets = [r + _TARGET_OFFSET for r in range(USER_MIN, USER_MAX + 1, 50)]
    gaps = [_gap_for_difficulty(t) for t in targets]
    assert all(a - b > 0.002 for a, b in pairwise(gaps)), "a band is indistinguishable"
    # And the aimable span is most of the axis, not a sliver of its middle.
    assert gaps[0] - gaps[-1] > 0.5


def test_the_curve_is_smooth_at_both_ends_of_the_evidence():
    """Linear where player strength constrains it, saturating past that at each
    end. Both joins have to be continuous in value *and* slope, or selection
    sees a cliff at an arbitrary gap and items either side are wrongly spaced."""
    for k in (CALIBRATED_GAP_LO, CALIBRATED_GAP_HI):
        assert difficulty_rating(k - 1e-9) == pytest.approx(difficulty_rating(k + 1e-9))
        below = (difficulty_rating(k) - difficulty_rating(k - 1e-4)) / 1e-4
        above = (difficulty_rating(k + 1e-4) - difficulty_rating(k)) / 1e-4
        assert below == pytest.approx(above, rel=1e-3)
        assert below == pytest.approx(-GAP_SLOPE, rel=1e-3)  # the measured slope


def test_the_shallow_gap_is_the_ladder_a_row_actually_stores():
    """`items.rating` is a pure function of `items.shallow_gap`, which is a pure
    function of `items.gap_ladder` — so the rounding has to happen before the
    difficulty does, or the stored rating is a function of a number no row keeps.
    """
    # Rungs that do not average to a stored-precision number, so dropping the
    # rounding is visible: the mean here runs to seventeen places.
    ladder = " ".join(f"{0.1 + i / 300:.4f}" for i in range(SHALLOW_PLIES))
    got = shallow_gap_of(ladder)
    assert got is not None and got == round(got, 4) == 0.1117
    # Short of the window is not a shallow gap at all: a mean over fewer rungs
    # would be a different measure wearing the same name.
    assert shallow_gap_of(" ".join(["0.1"] * (SHALLOW_PLIES - 1))) is None
    assert shallow_gap_of("") is None


def test_difficulty_matches_the_measured_slope_where_it_was_measured():
    """Across the calibrated band the curve is the empirical fit and nothing
    else, so a gap difference converts to rating points at the measured rate."""
    lo, hi = CALIBRATED_GAP_LO, CALIBRATED_GAP_HI
    measured = (difficulty_rating(lo) - difficulty_rating(hi)) / (hi - lo)
    assert measured == pytest.approx(GAP_SLOPE)


def test_every_user_on_the_scale_is_aimed_at_a_gap_an_item_can_have():
    """A target past either end of the curve hands a whole stretch of users the
    same clamped block, so the entire user range has to map to a real gap — and
    the map has to be monotone, or two users are aimed at each other's items."""
    gaps = [target_gap(r) for r in range(USER_MIN, USER_MAX, 50)]
    assert all(a > b for a, b in pairwise(gaps))
    # The bank's own range, so every user is aimed somewhere items exist.
    assert max(gaps) < 1.0 and min(gaps) > -1.0


def test_target_rating_hits_target_accuracy():
    # strip the jitter by averaging
    targets = [target_item_rating(1500) for _ in range(2000)]
    mean_target = sum(targets) / len(targets)
    assert math.isclose(expected_score(1500, mean_target), TARGET_ACCURACY, abs_tol=0.02)


def test_expected_score_bounds():
    assert expected_score(2000, 1000) > 0.95
    # Two alternatives, so nobody scores below the coin flip and the model may
    # not promise it: unfloored, Elo pays most of a K for every lucky guess on
    # an item far above the user — a rating pump with no information in it.
    assert expected_score(1000, 2000) == 0.5
    win, lose = update(1000, 2000, correct=True), update(1000, 2000, correct=False)
    assert win - 1000 == 1000 - lose  # a coin flip's wins and losses cancel


def test_the_easy_tail_decays_to_the_anchor_not_zero():
    """The curve's location is measured from live answers (RESPONSE_ANCHOR):
    users the old scale served at 80% expected were scoring 64%, uniformly, so
    the whole curve sits that much higher — including the easiest item there
    is, which is still a real distance up the user scale."""
    assert RESPONSE_ANCHOR < difficulty_rating(0.9) < RESPONSE_ANCHOR + 5
    assert difficulty_rating(CALIBRATED_GAP_HI) == pytest.approx(
        RESPONSE_ANCHOR + KNEE_DIFFICULTY  # the knee, moved without reshaping the tails
    )


def test_the_curve_inverts_cleanly_across_its_range():
    """`target_gap` is the inverse read of `difficulty_rating`, so the two have
    to agree everywhere an item can be — both tails included."""
    for g in (-0.4, -0.1, 0.02, 0.15, 0.27, 0.5):
        assert _gap_for_difficulty(difficulty_rating(g)) == pytest.approx(g, abs=1e-6)


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
