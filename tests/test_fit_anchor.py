"""The response-side fit recovers a planted scale offset.

Synthetic answers are generated under the deployed link with the item scale
shifted by a known amount; the fit has to hand the shift back — through the
recorded `calibrating` column and through the legacy delta inference alike,
since the live record holds both kinds of row.
"""

import numpy as np
import pytest

from trainer import db, rating
from trainer.fit_anchor import _expected, elo_scored, fit_anchor, load

PLANTED = 131.0


@pytest.fixture
def record(tmp_path):
    """A record of 150 calibrated users whose accuracy says the items really
    sit PLANTED points above what the model believed when it scored them."""
    conn = db.connect(tmp_path / "synthetic.db")
    conn.execute("PRAGMA foreign_keys=OFF")  # no real users or items behind these rows
    rng = np.random.default_rng(0)
    rows = []
    for user in range(1, 151):
        u = rng.uniform(500, 2600)
        for _ in range(3):  # a staircase prefix, which the fit must hold out
            rows.append((user, 1, "e2e4", 1, u, u - 237, u + 250.0, 0, 1))
        for _ in range(25):
            i = u - 237 + rng.uniform(-75, 75)
            correct = int(rng.random() < rating.expected_score(u, i + PLANTED))
            after = u + rating.K_USER * (correct - rating.expected_score(u, i))
            rows.append((user, 1, "e2e4", correct, u, i, after, 0, 0))
    conn.executemany(
        """INSERT INTO responses (user_id, item_id, choice_uci, correct,
             user_rating_before, item_rating_before, user_rating_after, shared,
             calibrating) VALUES (?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    conn.commit()
    return conn


def test_recovers_the_planted_offset(record):
    data = elo_scored(load(record))
    assert len(data) == 150 * 25  # every staircase row held out, nothing else
    assert fit_anchor(data) == pytest.approx(PLANTED, abs=25)


def test_legacy_rows_are_inferred_from_the_move_size(record):
    """Rows from before the `calibrating` column carry NULL, and the staircase
    has to be recognised by the one thing only it can do: move a rating by
    more than K_USER."""
    record.execute("UPDATE responses SET calibrating = NULL")
    record.commit()
    data = elo_scored(load(record))
    assert len(data) == 150 * 25
    assert fit_anchor(data) == pytest.approx(PLANTED, abs=25)


def test_the_fits_link_is_the_deployed_one():
    """`_expected` restates `rating.expected_score` for numpy; if the two ever
    disagree the fit measures a model the app doesn't run."""
    users = np.array([400.0, 850.0, 1500.0, 2200.0, 3000.0])
    items = np.array([1500.0, 131.0, 2900.0, 2000.0, 700.0])
    for u, i in zip(users, items, strict=True):
        assert _expected(np.array([u]), np.array([i]))[0] == pytest.approx(
            rating.expected_score(u, i)
        )
