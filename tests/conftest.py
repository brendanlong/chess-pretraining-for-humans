import threading

import pytest
from fastapi.testclient import TestClient

from trainer import auth, server
from trainer.db import connect
from trainer.rating import difficulty_rating

ITEM = {
    "best_uci": "e2e4",
    "distractor_uci": "a2a3",
    "distractor_source": "game",
    "cp_best": 50,
    "mate_best": None,
    "cp_distractor": -50,
    "mate_distractor": None,
    "wp_best": 0.55,
    "wp_distractor": 0.45,
    "gap_wp": 0.10,
    "learnable": 1,
    "depth_deep": 18,
    "depth_shallow": 8,
    # Derived, not chosen: `db.connect` re-derives `rating` from `gap_wp`, so a
    # fixture that picked its own would be rewritten out from under the test.
    "rating": difficulty_rating(0.10),
    "ply": 20,
    "game_url": "",
    "mover_elo": 1500,
    "time_control": "300+0",
}
# Distinct fens that all keep e2e4 and a2a3 legal.
FEN_TMPL = "rnbqkbnr/pppppppp/8/8/8/{}/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_RANKS = ["8", "7P", "6P1"]


def add_item(conn, fen: str, **overrides) -> None:
    """One item. `overrides` is for the tests that care what is in it — a bank
    with a difficulty distribution, say, rather than one of identical rows."""
    conn.execute(
        """INSERT INTO items (fen, best_uci, distractor_uci, distractor_source,
             cp_best, mate_best, cp_distractor, mate_distractor, wp_best,
             wp_distractor, gap_wp, learnable, depth_deep, depth_shallow,
             rating, ply, game_url, mover_elo, time_control)
           VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
             :cp_best, :mate_best, :cp_distractor, :mate_distractor, :wp_best,
             :wp_distractor, :gap_wp, :learnable, :depth_deep, :depth_shallow,
             :rating, :ply, :game_url, :mover_elo, :time_control)""",
        {**ITEM, "fen": fen, **overrides},
    )


@pytest.fixture
def item_count():
    """Override in a module/test to change how many items the bank holds."""
    return 2


class _NoDatabase:
    """Stands in for `server.conn` in tests that never asked for one."""

    def __getattr__(self, attribute):
        raise AssertionError(
            "this test reached server.conn without the `db` fixture, so it was "
            "about to use the real data/items.db — request `db` (or `client`) so "
            "it gets a throwaway database instead"
        )


@pytest.fixture(autouse=True)
def no_real_database(monkeypatch):
    """Make forgetting the `db` fixture loud instead of silent.

    `server.conn` is opened at import against the real database, so a test that
    hits the API without `db` quietly uses whatever the developer has in `data/` —
    passing on a laptop with a full item bank, failing in CI with an empty one,
    and writing rows into the experimental record on the way past. That is exactly
    how `test_a_rate_limit_can_say_what_it_is_actually_rationing` got committed.

    Autouse, so it applies first; `db` overrides it for the tests that want a
    database, and the one test that supplies its own connection still may.
    """
    monkeypatch.setattr(server, "conn", _NoDatabase())


@pytest.fixture
def db(tmp_path, monkeypatch, item_count):
    # TestClient runs the app in its own thread, like uvicorn's threadpool does.
    path = tmp_path / "test.db"
    conn = connect(path, check_same_thread=False)
    for i in range(item_count):
        add_item(conn, FEN_TMPL.format(FEN_RANKS[i]))
    conn.commit()
    # Point the server at this file rather than handing it this connection: the
    # server opens one per thread and each owns its own transaction, so a test
    # that shared a single connection with it would be exercising a concurrency
    # story the real thing doesn't have. The returned handle is a separate
    # connection to the same file, for tests that want to look at rows directly.
    monkeypatch.setattr(server, "DB_PATH", path)
    monkeypatch.setattr(server, "_threads", threading.local())
    monkeypatch.setattr(server, "conn", server._PerThreadConnection())
    # Fresh limiters per test, so one test's attempts can't starve another's.
    # (TestClient reports one host for everyone, so they all share a key.)
    for name, limiter in (
        ("signup_limiter", auth.RateLimiter(20, 3600)),
        ("login_limiter", auth.RateLimiter(20, 900)),
        ("login_ip_limiter", auth.RateLimiter(200, 900)),
        ("delete_limiter", auth.RateLimiter(20, 900)),
        # Every client in a test shares one address, and some tests answer in
        # bulk on purpose. The real limit gets its own test.
        ("answer_limiter", auth.RateLimiter(100_000, 900)),
        ("anonymous_trial_use", auth.RateLimiter(1, 900)),
    ):
        monkeypatch.setattr(server, name, limiter)
    return conn


@pytest.fixture
def client(db):
    """One TestClient is one browser: its cookie jar carries the session."""
    with TestClient(server.app) as c:
        yield c


def next_trial(client):
    r = client.get("/api/next")
    assert r.status_code == 200, r.text
    return r.json()


def answer(client, trial, choice: int = 0):
    r = client.post("/api/answer", json=answer_body(trial, choice))
    assert r.status_code == 200, r.text
    return r.json()


def answer_body(trial, choice: int = 0) -> dict:
    """What the client sends back. The token is the server's own proof that it
    offered this trial, so an answer without it isn't answering anything."""
    return {
        "item_id": trial["item_id"],
        "trial_token": trial["trial_token"],
        "choice_uci": trial["moves"][choice]["uci"],
    }
