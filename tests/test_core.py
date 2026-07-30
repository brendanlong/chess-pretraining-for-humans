import math

from trainer.label import MAX_GAP_WP, MIN_GAP_WP, difficulty_rating
from trainer.rating import (
    CALIB_END_STEP,
    CALIB_START_STEP,
    RATING_MAX,
    RATING_MIN,
    TARGET_ACCURACY,
    USER_MAX,
    USER_MIN,
    USER_START,
    calibrate,
    expected_score,
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


def test_difficulty_stays_on_the_rating_scale():
    """Difficulty is `gap_wp` mapped onto the user rating scale, so the widest
    and narrowest gaps the labeler admits still have to land inside it."""
    assert RATING_MIN <= difficulty_rating(MIN_GAP_WP) <= RATING_MAX
    assert RATING_MIN <= difficulty_rating(MAX_GAP_WP) <= RATING_MAX
    # A gap wide enough to run off the bottom of the scale is clamped onto it,
    # so `pick_item`'s nearest-rating search can always reach the easiest items.
    assert difficulty_rating(1.0) == RATING_MIN


def test_harder_items_rate_higher():
    assert difficulty_rating(0.02) > difficulty_rating(0.30)


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
