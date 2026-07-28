"""Centipawn -> win probability conversion.

Difficulty must be measured in win-probability space, not raw centipawns: a
1.0-pawn gap at eval 0.0 is enormous, while at +6.0 it is negligible. We use
the same logistic model Lichess uses for its accuracy metrics.
"""

import math

# Lichess's fitted constant (https://lichess.org/page/accuracy).
_K = 0.00368208
_CP_CLAMP = 1000


def cp_to_winprob(cp: float) -> float:
    """Win probability (0..1) for the side the cp score is relative to."""
    cp = max(-_CP_CLAMP, min(_CP_CLAMP, cp))
    return 1 / (1 + math.exp(-_K * cp))


def mate_to_cp(mate: int) -> int:
    """Map a mate-in-N score onto the clamped cp scale."""
    return _CP_CLAMP if mate > 0 else -_CP_CLAMP


def score_to_winprob(cp: int | None, mate: int | None) -> float:
    if mate is not None:
        return cp_to_winprob(mate_to_cp(mate))
    assert cp is not None
    return cp_to_winprob(cp)
