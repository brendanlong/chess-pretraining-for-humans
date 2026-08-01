"""Elo machinery for adaptive item selection.

An item's difficulty is a fixed property of the item: how far apart the two
moves are in win probability, and how far ahead you have to read to see that
they are, mapped onto the rating scale by `difficulty_rating` below. Only the
user has a rating that moves, so an answer is scored like a game against a
fixed opponent, and no two users are coupled through the bank.

That makes the user's rating a running estimate of the comparison they can
reliably see, and it makes each response independent of every response before
it — which is what the difficulty model in issue #27 needs, since it estimates
per-item difficulty offline where it can be regularised and where item
selection isn't feeding back into the thing being measured.

Items are selected so the user's expected score is ~TARGET_ACCURACY, the
perceptual-learning sweet spot: hard enough to carry signal, easy enough
that feedback stays mostly confirmatory.
"""

import math
import random
from itertools import pairwise

TARGET_ACCURACY = 0.80
K_USER = 32
SELECTION_JITTER = 75  # rating points of noise around the target difficulty
RATING_MIN = 0
# Above the hardest thing the curve can name: the hard tail saturates at
# HARD_CEILING, 2980. A guard, not a design feature — see `difficulty_rating`.
RATING_MAX = 3000

# How much of the ladder difficulty is a function of. Measured, not chosen: of
# every window 1..k, this is the one whose gap best predicts the strength of the
# players who got these positions wrong (a 326-point spread in the erring-elo
# tail, against 288 at one ply alone and 200 for the gap at full depth). Past
# about ten the window starts averaging in depths no human reaches and the
# signal falls away again. See `rating.GAP_SLOPE`.
SHALLOW_PLIES = 8


def shallow_gap_of(gap_ladder: str) -> float | None:
    """The stored ladder's first SHALLOW_PLIES rungs, averaged.

    Takes the stored text rather than the ladder object so that the one
    definition of "the shallow gap" serves the labeler, `db.connect`'s
    re-derivation and any later refit alike — and so that filling the column in
    on an existing bank needs no engine at all, only the ladder already there.

    None when there is no ladder, or when it is shorter than the window: a mean
    over fewer rungs would be a different measure wearing the same name.
    """
    if not gap_ladder:
        return None
    rungs = [float(x) for x in gap_ladder.split()[:SHALLOW_PLIES]]
    if len(rungs) < SHALLOW_PLIES:
        return None
    # Rounded like `gap_wp`, so the stored rating stays a pure function of the
    # stored column rather than of a number no row keeps.
    return round(sum(rungs) / len(rungs), 4)


# How much strength one point of win-probability gap is worth. Measured, not
# chosen: every 'game'-source item records the gap of a real error and the Elo
# of the human who made it, so binning those errors by player strength (100
# points wide, keeping bands of 120+ errors) and taking each band's
# 75th-percentile error size gives strength as a function of gap. Weighted by
# band size it comes out linear at 8899 rating points per unit gap, 95% CI
# [7719, 9632] over 400 bootstrap resamples.
#
# The 75th percentile rather than the largest error, because the *largest* is
# nearly strength-independent — everyone hangs a queen occasionally, and what
# separates strengths is how often. Frequency is unmeasurable here: mining only
# ever sees errors, never the moves that weren't errors, so there is no
# denominator. A quantile in from the tail is where the discrimination lives.
#
# Fitted on the untargeted half of the bank only. The other half was mined with
# `--min-gap-wp`/`--max-gap-wp` aimed at particular bands, which is deliberate
# selection on the very quantity being regressed: including it moves the same
# fit on the deep gap from 6096 to 17076. Run against the deep gap on the
# untargeted half this method returns 6101 against the 6096 that was published
# for it, which is the check that the method here is the method that produced
# that number.
GAP_SLOPE = 8899.0
# Player strength constrains the line only between these gaps — outside them
# there is no band of errors to fit against, because no strength of player has
# its typical error there.
CALIBRATED_GAP_LO = 0.07
CALIBRATED_GAP_HI = 0.27
# Outside the calibrated band the line would run away in both directions, so
# difficulty saturates instead, matching the line's value *and* slope at each
# join. Past the easy end it decays toward zero; past the hard end it rises
# toward a ceiling. Either way the gaps we know nothing about stay ordered and
# distinct rather than collapsing onto one clamped value, which is what SPEC
# requires — an unordered range is a range selection cannot aim inside.
#
# This is the one number here that was chosen rather than measured, and it is
# not cosmetic: it is the headroom the curve is allowed either side of the
# evidence, so it fixes where the scale sits *and* how fast both tails
# saturate, because the rate is the slope divided by it. Halving it squeezes
# every gap past 0.27 into half the room; doubling it pushes the hard end past
# RATING_MAX and back into a clamp. 600 keeps the whole bank inside the scale
# with each tail spread over 600 points. Anyone retuning it should check both
# ends.
KNEE_DIFFICULTY = 600.0
GAP_INTERCEPT = KNEE_DIFFICULTY + GAP_SLOPE * CALIBRATED_GAP_HI
# Where the hard tail saturates: the line's value at the hard edge of the
# evidence, plus the same headroom the easy tail gets.
HARD_CEILING = GAP_INTERCEPT - GAP_SLOPE * CALIBRATED_GAP_LO + KNEE_DIFFICULTY
_DECAY = GAP_SLOPE / KNEE_DIFFICULTY
# Rating minus target difficulty, at TARGET_ACCURACY. Negative: the item a user
# should be facing is easier than they are.
_TARGET_OFFSET = 400 * math.log10(1 / TARGET_ACCURACY - 1)


def gap_difficulty(shallow_gap: float) -> float:
    """The difficulty a shallow gap maps to, before any bounds are applied."""
    if shallow_gap > CALIBRATED_GAP_HI:
        return KNEE_DIFFICULTY * math.exp(-_DECAY * (shallow_gap - CALIBRATED_GAP_HI))
    if shallow_gap < CALIBRATED_GAP_LO:
        return HARD_CEILING - KNEE_DIFFICULTY * math.exp(
            -_DECAY * (CALIBRATED_GAP_LO - shallow_gap)
        )
    return GAP_INTERCEPT - GAP_SLOPE * shallow_gap


def _gap_for_difficulty(difficulty: float) -> float:
    """Inverse of `difficulty_rating`, for reading a scale position as a gap."""
    if difficulty <= KNEE_DIFFICULTY:
        # Clamped at the bottom: a difficulty at or below zero has no gap behind
        # it, and the easy tail never reaches zero however wide the gap.
        return CALIBRATED_GAP_HI - math.log(max(difficulty, 1e-9) / KNEE_DIFFICULTY) / _DECAY
    if difficulty >= HARD_CEILING - KNEE_DIFFICULTY:
        # Symmetrically at the top: the hard tail approaches HARD_CEILING and
        # never arrives, so a difficulty at or past it reads as the hardest gap
        # the curve can name rather than as a negative one.
        room = max(HARD_CEILING - difficulty, 1e-9)
        return CALIBRATED_GAP_LO + math.log(room / KNEE_DIFFICULTY) / _DECAY
    return (GAP_INTERCEPT - difficulty) / GAP_SLOPE


def difficulty_rating(shallow_gap: float) -> float:
    """An item's difficulty, in the same units as a user's rating.

    `shallow_gap` is the item's win-probability gap as the shallow end of the
    engine's own search saw it — `label.SHALLOW_PLIES` rungs of the ladder,
    averaged — and not the gap at full depth. That is the whole point: the deep
    gap is what the answer is worth, while this is what there was to see, and
    measured against the strength of the humans who got it wrong the second
    predicts about one and a half times as much as the first. Everything the
    old lookahead-depth axis carried is in here too, continuously: a gap that
    is wide early is easy, one that is narrow is hard, and one that is
    *negative* is an item whose surface actively recommends the wrong move.

    Lives here rather than in the labeler that first applies it because it is
    the definition of `items.rating`, which `db.connect` re-derives and the
    selection query compares against user ratings. Callers must pass the
    `shallow_gap` that gets *stored*, not the full-precision one it was rounded
    from, or the stored rating stops being a function of the stored column.
    """
    # Bounds are a guard, not a design feature: the curve is already inside them
    # for every gap there is, and clamping would flatten an end of the bank.
    return max(RATING_MIN, min(RATING_MAX, gap_difficulty(shallow_gap)))


# New users start as "knows the rules but is terrible" and calibrate upward,
# rather than starting mid-scale and asking. Plain Elo can't climb fast from
# a too-low start (a strong player beating easy items gains almost nothing
# per win), so new users get a staircase: big jumps while they keep winning,
# step halves on each miss, hand off to Elo once the step is small.
USER_START = 850
CALIB_START_STEP = 250
CALIB_END_STEP = 40  # below this, calibration is over
USER_MIN = 350
# The top of the item scale is what a user can be aimed at, so the user scale
# stops where its target reaches the hardest items that exist. Past that the
# inverse of the curve runs off into gaps no position has, and every user up
# there is aimed at the same handful.
USER_MAX = 3200


def target_gap(user_rating: float) -> float:
    """The shallow win-probability gap a user of this rating is aimed at."""
    return _gap_for_difficulty(user_rating + _TARGET_OFFSET)


# The deep-gap curve, kept only because ratings written under it still exist.
# `regraded_user_rating` is the one thing that still needs to know it.
_OLD_SLOPE, _OLD_KNEE, _OLD_HI = 6096.0, 600.0, 0.33
_OLD_INTERCEPT = _OLD_KNEE + _OLD_SLOPE * _OLD_HI


def _deep_gap_difficulty(gap_wp: float) -> float:
    if gap_wp <= _OLD_HI:
        return _OLD_INTERCEPT - _OLD_SLOPE * gap_wp
    return _OLD_KNEE * math.exp(-(_OLD_SLOPE / _OLD_KNEE) * (gap_wp - _OLD_HI))


def _precurve_to_deep(old_rating: float) -> float:
    """The first regrade: off the flat pre-curve scale onto the deep-gap one.

    Reads the gap the old scale aimed at and returns the rating aimed at the
    same gap on the deep-gap curve, which is what it preserved.
    """
    gap = (2400 - (old_rating + _TARGET_OFFSET)) / 5000
    return _deep_gap_difficulty(gap) - _TARGET_OFFSET


# How a rating on the deep-gap scale reads on the shallow-gap one. Anchors at
# every 200 points, interpolated between; outside them the ends are held.
#
# What a regrade has to preserve is what a user is *served*, and these two
# scales have no gap in common to preserve — they are readings of different
# searches. What they do share is the bank, so each anchor is the rating whose
# target sits at the same percentile of item difficulty as the old one did.
# Nobody's trials change on the day it runs; every displayed number does, by
# about 700 in the middle, because the new axis is intrinsically taller. Its
# calibrated band alone spans 1780 rating points against the old one's 1341, so
# no choice of constant would have lowered it to meet the old scale — doing that
# would have pushed the easy tail below zero.
#
# Anchors rather than a formula because the map is the shape of the bank's
# difficulty distribution and is not linear: a straight line through these is
# out by up to 361 points, which is five selection jitters.
_DEEP_TO_SHALLOW = (
    (300, 359),
    (500, 683),
    (700, 1122),
    (900, 1489),
    (1100, 1802),
    (1300, 2012),
    (1500, 2200),
    (1700, 2371),
    (1900, 2546),
    (2100, 2686),
    (2300, 2795),
    (2500, 2907),
    (2700, 3074),
)


def _interpolate(anchors: tuple[tuple[float, float], ...], x: float) -> float:
    """Piecewise-linear through `anchors`, held flat outside them."""
    if x <= anchors[0][0]:
        return anchors[0][1]
    if x >= anchors[-1][0]:
        return anchors[-1][1]
    for (x0, y0), (x1, y1) in pairwise(anchors):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise AssertionError("anchors must be sorted")


def regraded_user_rating(old_rating: float, from_version: int) -> float:
    """Move a rating from the scale `from_version` wrote onto the current one.

    Chained rather than switched, because a database restored from far enough
    back is two scales behind and has to cross both. A rating carries no mark of
    which scale produced it, so `meta.schema_version` is the only thing that
    knows, which is why the migration that applies this is gated on it.
    """
    if from_version < 1:
        old_rating = _precurve_to_deep(old_rating)
    if from_version < 2:
        old_rating = _interpolate(_DEEP_TO_SHALLOW, old_rating)
    return max(USER_MIN, min(USER_MAX, old_rating))


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
    """The user's new rating after one answer against a fixed-difficulty item.

    Bounded like the staircase is, because Elo on its own has no floor: a run
    of misses walks a rating off the bottom of the scale, and past the end of
    it every target picks out the same easiest handful of items the bank has.
    """
    s = 1.0 if correct else 0.0
    moved = user_rating + K_USER * (s - expected_score(user_rating, item_rating))
    return max(USER_MIN, min(USER_MAX, moved))
