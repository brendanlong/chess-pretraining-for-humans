import pytest
from fastapi.testclient import TestClient

from trainer import auth, server
from trainer.db import connect

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
    "rating": 1500,
    "ply": 20,
    "game_url": "",
    "mover_elo": 1500,
    "time_control": "300+0",
}
# Distinct fens that all keep e2e4 and a2a3 legal.
FEN_TMPL = "rnbqkbnr/pppppppp/8/8/8/{}/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
FEN_RANKS = ["8", "7P", "6P1"]


def add_item(conn, fen: str) -> None:
    conn.execute(
        """INSERT INTO items (fen, best_uci, distractor_uci, distractor_source,
             cp_best, mate_best, cp_distractor, mate_distractor, wp_best,
             wp_distractor, gap_wp, learnable, depth_deep, depth_shallow,
             rating, ply, game_url, mover_elo, time_control)
           VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
             :cp_best, :mate_best, :cp_distractor, :mate_distractor, :wp_best,
             :wp_distractor, :gap_wp, :learnable, :depth_deep, :depth_shallow,
             :rating, :ply, :game_url, :mover_elo, :time_control)""",
        {**ITEM, "fen": fen},
    )


@pytest.fixture
def item_count():
    """Override in a module/test to change how many items the bank holds."""
    return 2


@pytest.fixture
def db(tmp_path, monkeypatch, item_count):
    # TestClient runs the app in its own thread, like uvicorn's threadpool does.
    conn = connect(tmp_path / "test.db", check_same_thread=False)
    for i in range(item_count):
        add_item(conn, FEN_TMPL.format(FEN_RANKS[i]))
    conn.commit()
    monkeypatch.setattr(server, "conn", conn)
    # Fresh limiters per test, so one test's attempts can't starve another's.
    # (TestClient reports one host for everyone, so they all share a key.)
    for name, limiter in (
        ("signup_limiter", auth.RateLimiter(20, 3600)),
        ("login_limiter", auth.RateLimiter(20, 900)),
        ("login_ip_limiter", auth.RateLimiter(200, 900)),
        ("delete_limiter", auth.RateLimiter(20, 900)),
        # Every client in a test shares one address, and some tests mint
        # hundreds of guests on purpose. The real limit gets its own test.
        ("guest_limiter", auth.RateLimiter(100_000, 3600)),
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


def answer(client, trial):
    r = client.post(
        "/api/answer",
        json={"item_id": trial["item_id"], "choice_uci": trial["moves"][0]["uci"]},
    )
    assert r.status_code == 200, r.text
    return r.json()
