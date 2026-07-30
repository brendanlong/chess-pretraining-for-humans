"""Elo machinery for adaptive item selection.

An item's difficulty is a fixed property of the item: its win-probability gap,
mapped onto the rating scale by `label.difficulty_rating`. Only the user has a
rating that moves, so an answer is scored like a game against a fixed opponent,
and no two users are coupled through the bank.

That makes the user's rating a running estimate of the gap they can reliably
see, and it makes each response independent of every response before it —
which is what the difficulty model in issue #27 needs, since it estimates
per-item difficulty offline where it can be regularised and where item
selection isn't feeding back into the thing being measured.

Items are selected so the user's expected score is ~TARGET_ACCURACY, the
perceptual-learning sweet spot: hard enough to carry signal, easy enough
that feedback stays mostly confirmatory.
"""

import math
import random

TARGET_ACCURACY = 0.80
K_USER = 32
SELECTION_JITTER = 75  # rating points of noise around the target difficulty
RATING_MIN = 0
RATING_MAX = 2600

# How much strength one point of win-probability gap is worth. Measured, not
# chosen: every 'game'-source item records the gap of a real error and the Elo
# of the human who made it, so binning ~12.5k errors by player strength and
# taking each band's 75th-percentile error size gives strength as a function of
# gap. It comes out linear at -5893 rating points per unit gap (95% CI
# [-4845, -6142] over 400 bootstrap resamples; AIC and leave-one-band-out CV
# both prefer linear over log and logistic forms).
#
# The 75th percentile rather than the largest error, because the *largest* is
# nearly strength-independent — everyone hangs a queen occasionally, and what
# separates strengths is how often. Frequency is unmeasurable here: mining only
# ever sees errors, never the moves that weren't errors, so there is no
# denominator. A quantile in from the tail is where the discrimination lives.
GAP_SLOPE = 5893.0
GAP_INTERCEPT = 2538.0  # placement on the scale; a uniform shift changes nothing
# Player strength only constrains the fit between these gaps: below it the bank
# is thin, and above it errors are rare at *every* strength (even 850-rated
# players have a 75th-percentile error of 0.325), so nothing in the data speaks
# to how much easier a 0.5 gap is than a 0.4 one.
CALIBRATED_GAP_LO = 0.11
CALIBRATED_GAP_HI = 0.33
# Past that the line would cross zero and take a third of the bank with it, so
# difficulty decays toward zero instead — same value and same slope at the
# knee, so the curve stays smooth, and gaps we know nothing about stay ordered
# and distinct rather than collapsing onto one clamped value.
_KNEE = GAP_INTERCEPT - GAP_SLOPE * CALIBRATED_GAP_HI
_DECAY = GAP_SLOPE / _KNEE
# Rating minus target difficulty, at TARGET_ACCURACY. Negative: the item a user
# should be facing is easier than they are.
_TARGET_OFFSET = 400 * math.log10(1 / TARGET_ACCURACY - 1)


def _gap_for_difficulty(difficulty: float) -> float:
    """Inverse of `difficulty_rating`, for reading a scale position as a gap."""
    if difficulty >= _KNEE:
        return (GAP_INTERCEPT - difficulty) / GAP_SLOPE
    return CALIBRATED_GAP_HI - math.log(max(difficulty, 1e-9) / _KNEE) / _DECAY


def difficulty_rating(gap_wp: float) -> float:
    """An item's difficulty, in the same units as a user's rating.

    Lives here rather than in the labeler that first applies it because it is
    the definition of `items.rating`, which `db.connect` re-derives and the
    selection query compares against user ratings. Callers must pass the
    `gap_wp` that gets *stored*, not the full-precision one it was rounded
    from, or the stored rating stops being a function of the stored gap.
    """
    if gap_wp <= CALIBRATED_GAP_HI:
        d = GAP_INTERCEPT - GAP_SLOPE * gap_wp
    else:
        d = _KNEE * math.exp(-_DECAY * (gap_wp - CALIBRATED_GAP_HI))
    # Bounds are a guard, not a design feature: the curve is already inside
    # them for every gap the labeler admits, and it is the clamping that used
    # to happen here that flattened the easy end of the bank.
    return max(RATING_MIN, min(RATING_MAX, d))


# New users start as "knows the rules but is terrible" and calibrate upward,
# rather than starting mid-scale and asking. Plain Elo can't climb fast from
# a too-low start (a strong player beating easy items gains almost nothing
# per win), so new users get a staircase: big jumps while they keep winning,
# step halves on each miss, hand off to Elo once the step is small.
USER_START = 575
CALIB_START_STEP = 250
CALIB_END_STEP = 40  # below this, calibration is over
USER_MIN = 300
USER_MAX = 2800


def target_gap(user_rating: float) -> float:
    """The win-probability gap a user of this rating is aimed at."""
    return _gap_for_difficulty(user_rating + _TARGET_OFFSET)


def regraded_user_rating(old_rating: float) -> float:
    """A rating from the pre-curve scale, moved onto the current one.

    A rating only means anything against the difficulty it selects, so what a
    regrade has to preserve is the gap a user is served, not the number. This
    reads the gap the old scale aimed them at and returns the rating aimed at
    the same gap now, which is why nobody's trials change on the day it runs.

    The old constants live here because rows written under them still exist;
    this is the only thing that still needs to know them.
    """
    old_target = old_rating + _TARGET_OFFSET
    gap = (2400 - old_target) / 5000
    return max(USER_MIN, min(USER_MAX, difficulty_rating(gap) - _TARGET_OFFSET))


def calibrate(user_rating: float, step: float, correct: bool) -> tuple[float, float]:
    """One staircase move; returns (new_rating, new_step)."""
    if correct:
        new = user_rating + step
    else:
        new = user_rating - step
        step /= 2
    return max(USER_MIN, min(USER_MAX, new)), step


def expected_score(user_rating: float, item_rating: float) -> float:
    return 1 / (1 + 10 ** ((item_rating - user_rating) / 400))


def target_item_rating(user_rating: float) -> float:
    """Item rating at which the user's expected score is TARGET_ACCURACY."""
    return user_rating + _TARGET_OFFSET + random.uniform(-SELECTION_JITTER, SELECTION_JITTER)


def update(user_rating: float, item_rating: float, correct: bool) -> float:
    """The user's new rating after one answer against a fixed-difficulty item."""
    s = 1.0 if correct else 0.0
    return user_rating + K_USER * (s - expected_score(user_rating, item_rating))
