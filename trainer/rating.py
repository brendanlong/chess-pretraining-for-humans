"""Elo machinery for adaptive item selection.

Every item carries a difficulty rating seeded from its win-probability gap
(see label.py) and updated from real responses, because gap alone can't hold
a target accuracy — an obvious hanging-piece capture and a subtle
prophylactic move can share a gap. The user has a rating too; each answer is
scored like a game between user and item.

Items are selected so the user's expected score is ~TARGET_ACCURACY, the
perceptual-learning sweet spot: hard enough to carry signal, easy enough
that feedback stays mostly confirmatory.
"""

import math
import random

TARGET_ACCURACY = 0.80
K_USER = 32
K_ITEM = 16
SELECTION_JITTER = 75  # rating points of noise around the target difficulty
RATING_MIN = 600
RATING_MAX = 2500

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


def update(user_rating: float, item_rating: float, correct: bool) -> tuple[float, float]:
    e = expected_score(user_rating, item_rating)
    s = 1.0 if correct else 0.0
    new_item = item_rating - K_ITEM * (s - e)
    # Items stay inside the seed-prior range so a streak on a rarely-served
    # item can't drift it out of selection reach.
    new_item = max(RATING_MIN, min(RATING_MAX, new_item))
    return user_rating + K_USER * (s - e), new_item
