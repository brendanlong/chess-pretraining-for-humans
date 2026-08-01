"""Reading a required lookahead off a ladder of paired engine evaluations.

The search itself needs Stockfish and is not exercised here; what is, is the
rule that turns its output into a number — which is where every judgement call
in the measurement lives.
"""

from trainer.label import shallowest_settled


def ladder(*rungs: tuple[float, float]) -> dict[int, tuple[float, float]]:
    """Depths 1..n, each a (best, distractor) pair as the search ranked them."""
    return dict(enumerate(rungs, start=1))


def test_a_refutation_visible_at_once_reads_as_one_ply():
    assert shallowest_settled(ladder((0.6, 0.4), (0.6, 0.4), (0.6, 0.4))) == 1


def test_the_depth_the_ordering_arrives_at_is_the_answer():
    assert shallowest_settled(ladder((0.4, 0.5), (0.4, 0.5), (0.6, 0.4), (0.6, 0.4))) == 3


def test_an_ordering_that_flips_back_is_not_settled_where_it_first_looked_right():
    """The reason the walk goes down from the deepest rung. A comparison that
    happens to look right at one ply and wrong at the next was not seen at one
    ply — calling it easy would be reading a coincidence as a perception."""
    assert shallowest_settled(ladder((0.6, 0.4), (0.4, 0.5), (0.6, 0.4), (0.6, 0.4))) == 3


def test_a_comparison_the_deepest_search_gets_wrong_has_no_depth():
    """Not "too hard to serve" — the scale has room for the hardest thing the
    engine can see. It means the deep pass and a search restricted to the pair
    disagree about the answer, so there is nothing to teach."""
    assert shallowest_settled(ladder((0.6, 0.4), (0.6, 0.4), (0.4, 0.5))) is None


def test_a_tie_is_not_seeing_it():
    """Equal evaluations order the two moves no better than a coin does."""
    assert shallowest_settled(ladder((0.5, 0.5), (0.6, 0.4))) == 2


def test_a_search_that_stopped_early_is_judged_where_it_stopped():
    """Finding a mate ends the iteration, so a ladder can be short. Its deepest
    rung is the deepest evidence there is."""
    assert shallowest_settled(ladder((0.9, 0.5), (0.99, 0.5))) == 1


def test_depths_are_read_in_order_and_not_as_they_arrived():
    """The rungs come off a stream, so a dict here is insertion-ordered by when
    the engine emitted each one. Walking that order rather than sorting it would
    make the answer depend on the order Stockfish happened to report."""
    scrambled = {3: (0.6, 0.4), 1: (0.4, 0.5), 2: (0.6, 0.4)}
    assert shallowest_settled(scrambled) == 2


def test_an_empty_ladder_is_no_answer():
    assert shallowest_settled({}) is None
