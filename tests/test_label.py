"""Reading a required lookahead off two ladders of engine evaluations.

The searches themselves need Stockfish and are not exercised here; what is,
is the rule that turns their output into a number — which is where every
judgement call in the measurement lives.
"""

from trainer.label import shallowest_settled


def ladder(*wps: float) -> dict[int, float]:
    """Depths 1..n, in order."""
    return dict(enumerate(wps, start=1))


def test_a_refutation_visible_at_once_reads_as_one_ply():
    assert shallowest_settled(ladder(0.6, 0.6, 0.6), ladder(0.4, 0.4, 0.4)) == 1


def test_the_depth_the_ordering_arrives_at_is_the_answer():
    assert shallowest_settled(ladder(0.4, 0.4, 0.6, 0.6), ladder(0.5, 0.5, 0.4, 0.4)) == 3


def test_an_ordering_that_flips_back_is_not_settled_where_it_first_looked_right():
    """The reason the walk goes down from the deepest rung. A comparison that
    happens to look right at one ply and wrong at the next was not seen at one
    ply — calling it easy would be reading a coincidence as a perception."""
    assert shallowest_settled(ladder(0.6, 0.4, 0.6, 0.6), ladder(0.4, 0.5, 0.4, 0.4)) == 3


def test_a_comparison_no_search_gets_right_has_no_depth():
    """Which is the learnability filter: not reachable from the surface."""
    assert shallowest_settled(ladder(0.4, 0.4, 0.4), ladder(0.5, 0.5, 0.5)) is None


def test_a_tie_is_not_seeing_it():
    """Equal evaluations order the two moves no better than a coin does."""
    assert shallowest_settled(ladder(0.5, 0.6), ladder(0.5, 0.4)) == 2


def test_a_rung_only_one_search_reported_is_stepped_over():
    """A missing depth is no evidence either way. Ending the walk there would
    report every comparison as needing more lookahead than it does."""
    best = {1: 0.6, 3: 0.6}
    assert shallowest_settled(best, ladder(0.4, 0.4, 0.4)) == 1


def test_a_search_that_stopped_early_is_judged_where_it_stopped():
    """Finding a mate ends the iteration, so the ladders can be short. The
    deepest rung they share is the deepest evidence there is."""
    assert shallowest_settled(ladder(0.9, 0.99), ladder(0.5, 0.5, 0.5, 0.5)) == 1


def test_no_shared_rung_is_no_answer():
    assert shallowest_settled({1: 0.6}, {2: 0.4}) is None
    assert shallowest_settled({}, {}) is None
