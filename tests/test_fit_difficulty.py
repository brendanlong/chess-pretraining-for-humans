"""The arithmetic behind the difficulty constants.

Worth its own tests because it is the thing that decides whether a published
constant still matches the rows — a fit that quietly drifts turns a
re-measurement into an argument nobody can settle. The synthetic samples here
are built to a known boundary, so the fit has an answer to be wrong about;
`CALIBRATION.md` is where the real ones are checked against the bank.
"""

import math

import numpy as np
import pytest

from trainer.fit_difficulty import fit, spread, window

BOUNDARY = 5000  # rating points per unit gap, in the samples below


def erring_players(rng, per_band=40, step=5):
    """Errors from a world where a player of strength `elo` makes mistakes of
    up to (2600 - elo) / BOUNDARY and no larger. The 75th percentile per band
    then traces that line, so `fit` has to return BOUNDARY."""
    elo = np.repeat(np.arange(600, 2500, step, dtype=float), per_band)
    ceiling = (2600 - elo) / BOUNDARY
    return elo, rng.uniform(0, ceiling / 0.75)


def test_fit_recovers_a_slope_it_was_given():
    slope, bands = fit(*erring_players(np.random.default_rng(0)))
    assert bands == 19
    assert slope == pytest.approx(BOUNDARY, rel=0.05)


def test_fit_reads_the_seventy_fifth_percentile_and_not_the_point_seven_fifth():
    """`QUANTILE` is a fraction, so `np.percentile` — which takes 0..100 — is
    the wrong function and answers a different question. It does not raise; it
    fits a plausible-looking slope off the very bottom of each band, which is
    how a silently wrong constant would get published."""
    elo, gap = erring_players(np.random.default_rng(1))
    assert fit(elo, gap)[0] == pytest.approx(BOUNDARY, rel=0.05)
    # What the mistake would have looked like: a quantile a hundred times too
    # low reads a different part of every band and lands nowhere near.
    bottom = np.quantile(gap[elo == 1000], 0.0075)
    assert bottom < np.quantile(gap[elo == 1000], 0.75) / 10


def test_fit_says_nothing_rather_than_inventing_a_slope():
    """Too few bands, or every band identical, are both "this sample can't say".
    A traceback would be a worse way to report that than a nan that prints."""
    one_band = fit(np.full(500, 1500.0), np.full(500, 0.2))
    assert math.isnan(one_band[0]) and one_band[1] == 1
    elo = np.repeat(np.arange(600, 2500, 5, dtype=float), 40)
    flat = fit(elo, np.full(elo.size, 0.2))
    assert math.isnan(flat[0]) and flat[1] == 19


def test_a_thin_band_is_dropped_rather_than_allowed_to_swing_the_fit():
    """One band of a handful of errors has no stable quantile in it, and it
    would sit at the far end of the x range where it drags hardest."""
    elo, gap = erring_players(np.random.default_rng(3))
    strays = (np.full(5, 3000.0), np.full(5, 0.9))
    assert fit(np.concatenate([elo, strays[0]]), np.concatenate([gap, strays[1]])) == fit(elo, gap)


def test_spread_scores_a_measure_that_orders_strength_above_one_that_doesnt():
    """The comparison every candidate axis is judged on, so it has to actually
    reward ordering — a measure uncorrelated with strength must score near
    zero, and one that tracks it must not."""
    rng = np.random.default_rng(1)
    elo = rng.uniform(800, 2400, 3000)
    assert spread(elo, elo + rng.normal(0, 50, 3000)) > 1000
    assert abs(spread(elo, rng.uniform(0, 1, 3000))) < 150


def test_a_window_shorter_than_the_ladder_falls_back_to_its_last_rung():
    """Short ladders don't reach the fit — `rows` filters them — but the helper
    is the one place that would silently average a different number of rungs
    for different items if it didn't say what it does."""
    assert window([0.1, 0.2, 0.3, 0.4], 2) == pytest.approx(0.15)
    assert window([0.1, 0.2], 8) == 0.2
