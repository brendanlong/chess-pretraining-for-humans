"""The arithmetic behind the difficulty constants.

Worth its own tests because it is the thing that decides whether a published
constant still matches the rows — a fit that quietly drifts turns a
re-measurement into an argument nobody can settle.
"""

import math
import random

import pytest

from trainer.fit_difficulty import fit, quantile, spread, window


def test_quantile_interpolates_between_order_statistics():
    """numpy's default, and so the published constants'. A nearest-rank
    quantile is a different estimator and moves the fitted slope by percent,
    which would read as a disagreement rather than as a convention."""
    values = [0.0, 1.0, 2.0, 3.0]
    assert quantile(values, 0.0) == 0.0
    assert quantile(values, 1.0) == 3.0
    assert quantile(values, 0.5) == 1.5
    assert quantile(values, 0.75) == 2.25  # nearest-rank would say 3.0
    assert quantile([7.0], 0.75) == 7.0


def test_fit_recovers_a_slope_it_was_given():
    """Synthetic errors built to a known boundary: a player of strength `elo`
    makes errors up to (2600 - elo) / 5000 and no larger, so the 75th
    percentile per band traces that line and the fit has to return 5000."""
    rng = random.Random(0)
    sample = []
    for elo in range(600, 2500, 5):
        ceiling = (2600 - elo) / 5000
        sample += [(float(elo), rng.uniform(0, ceiling / 0.75)) for _ in range(40)]
    slope, bands = fit(sample)
    assert bands == 19
    assert slope == pytest.approx(5000, rel=0.05)


def test_fit_says_nothing_rather_than_inventing_a_slope():
    """Too few bands, or every band identical, are both "this sample can't say".
    A traceback would be a worse way to report that than a nan that prints."""
    one_band = fit([(1500.0, 0.2)] * 500)
    assert math.isnan(one_band[0]) and one_band[1] == 1
    flat = fit([(float(e), 0.2) for e in range(600, 2500, 5) for _ in range(40)])
    assert math.isnan(flat[0]) and flat[1] == 19


def test_a_thin_band_is_dropped_rather_than_allowed_to_swing_the_fit():
    """One band of a handful of errors has no stable quantile in it, and it
    would sit at the far end of the x range where it drags hardest."""
    rng = random.Random(3)
    dense = [
        (float(e), rng.uniform(0, (2600 - e) / 5000 / 0.75))
        for e in range(600, 2500, 5)
        for _ in range(40)
    ]
    assert fit(dense + [(3000.0, 0.9)] * 5) == fit(dense)


def test_spread_scores_a_measure_that_orders_strength_above_one_that_doesnt():
    """The comparison every candidate axis is judged on, so it has to actually
    reward ordering — a measure uncorrelated with strength must score near
    zero, and one that tracks it must not."""
    rng = random.Random(1)
    elos = [rng.uniform(800, 2400) for _ in range(3000)]
    tracks = [(e, e + rng.gauss(0, 50)) for e in elos]
    noise = [(e, rng.uniform(0, 1)) for e in elos]
    assert spread(tracks) > 1000
    assert abs(spread(noise)) < 150


def test_a_window_shorter_than_the_ladder_falls_back_to_its_last_rung():
    """Short ladders don't reach the fit — `rows` filters them — but the helper
    is the one place that would silently average a different number of rungs
    for different items if it didn't say what it does."""
    assert window([0.1, 0.2, 0.3, 0.4], 2) == pytest.approx(0.15)
    assert window([0.1, 0.2], 8) == 0.2
