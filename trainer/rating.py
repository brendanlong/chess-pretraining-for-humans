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
RATING_MIN = 600
RATING_MAX = 2500


def difficulty_rating(gap_wp: float) -> float:
    """An item's difficulty: a 2% win-prob gap is expert-hard, 35% is trivial.

    Lives here rather than in the labeler that first applies it because it is
    the definition of `items.rating`, which `db.connect` re-derives and the
    selection query compares against user ratings. Callers must pass the
    `gap_wp` that gets *stored*, not the full-precision one it was rounded
    from, or the stored rating stops being a function of the stored gap.
    """
    return max(RATING_MIN, min(RATING_MAX, 2400 - 5000 * gap_wp))


# New users start as "knows the rules but is terrible" and calibrate upward,
# rather than starting mid-scale and asking. Plain Elo can't climb fast from
# a too-low start (a strong player beating easy items gains almost nothing
# per win), so new users get a staircase: big jumps while they keep winning,
# step halves on each miss, hand off to Elo once the step is small.
USER_START = 700
CALIB_START_STEP = 250
CALIB_END_STEP = 40  # below this, calibration is over
USER_MIN = 400
USER_MAX = 2600


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
    offset = 400 * math.log10(1 / TARGET_ACCURACY - 1)
    return user_rating + offset + random.uniform(-SELECTION_JITTER, SELECTION_JITTER)


def update(user_rating: float, item_rating: float, correct: bool) -> float:
    """The user's new rating after one answer against a fixed-difficulty item."""
    s = 1.0 if correct else 0.0
    return user_rating + K_USER * (s - expected_score(user_rating, item_rating))
