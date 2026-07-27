import pytest

from trainer import server
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
START_FEN_TMPL = "rnbqkbnr/pppppppp/8/8/8/{}/PPPPPPPP/RNBQKBNR w KQkq - 0 1"


@pytest.fixture
def two_item_db(tmp_path, monkeypatch):
    conn = connect(tmp_path / "test.db")
    for i in range(2):
        # distinct fens; both keep e2e4 and a2a3 legal
        fen = START_FEN_TMPL.format("8" if i == 0 else "7P")
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
    conn.commit()
    monkeypatch.setattr(server, "conn", conn)
    return conn


def answer(trial, user):
    return server.answer(
        server.Answer(item_id=trial["item_id"], choice_uci=trial["moves"][0]["uci"], user=user)
    )


def test_no_repeats_until_exhausted_then_flagged(two_item_db):
    seen = set()
    for _ in range(2):
        t = server.next_item(user="u")
        assert t["repeat"] is False
        assert t["item_id"] not in seen
        seen.add(t["item_id"])
        result = answer(t, "u")
        assert result["repeat"] is False
        assert "correct" in result and "best" in result  # feedback on every trial

    # bank exhausted: repeats are flagged and rating-inert
    t = server.next_item(user="u")
    assert t["repeat"] is True
    assert t["items_remaining"] == 0
    rating_before = server.get_user("u")["rating"]
    result = answer(t, "u")
    assert result["repeat"] is True
    assert "correct" in result  # feedback still shown
    assert server.get_user("u")["rating"] == rating_before  # but no rating movement


def test_first_exposure_accuracy_excludes_repeats(two_item_db):
    for _ in range(4):  # 2 fresh + 2 repeats
        answer(server.next_item(user="u"), "u")
    stats = server.stats(user="u")
    assert stats["attempts"] == 4
    assert stats["first_exposures"] == 2
    assert len(stats["rating_history"]) == 2
