import chess
import chess.engine
import chess.pgn

from trainer.mine import MIN_PLY, base_time_seconds, mine_game, raw_games, score_cp

GAME_A = '[Event "A"]\n[Site "a"]\n\n1. e4 e5 1/2-1/2\n\n'
GAME_B = '[Event "B"]\n[Site "b"]\n\n1. d4 d5 1/2-1/2\n\n'

# Ruy Lopez, long enough that a candidate can land past the opening-book cutoff.
MOVES = [
    "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4",
    "g8f6", "e1g1", "f8e7", "f1e1", "b7b5", "a4b3", "d7d6",
]  # fmt: skip


def test_raw_games_splits_on_the_blank_line_pairs():
    assert list(raw_games(iter((GAME_A + GAME_B).splitlines(keepends=True)))) == [GAME_A, GAME_B]


def test_raw_games_drops_a_truncated_trailing_game():
    # streaming the head of a dump cuts mid-game; that fragment must not be yielded
    truncated = GAME_A + '[Event "B"]\n[Site "b"]\n\n1. d4'
    assert list(raw_games(iter(truncated.splitlines(keepends=True)))) == [GAME_A]


def test_base_time_seconds_handles_unrated_and_correspondence():
    assert base_time_seconds(chess.pgn.Headers(TimeControl="300+3")) == 300
    assert base_time_seconds(chess.pgn.Headers(TimeControl="-")) == 0


def test_score_cp_skips_mates():
    assert score_cp(chess.engine.Cp(-42)) == -42
    assert score_cp(chess.engine.Mate(3)) is None


def annotated_game(blunder_cp: int, ply: int = MIN_PLY, time_control: str = "300+0"):
    """A game level at 0.00 until the move at `ply` drops it to blunder_cp."""
    game = chess.pgn.Game()
    game.headers.update(
        {"Site": "https://lichess.org/x", "TimeControl": time_control, "WhiteElo": "1800"}
    )
    node = game
    for i, uci in enumerate(MOVES):
        node = node.add_main_variation(chess.Move.from_uci(uci))
        cp = blunder_cp if i >= ply else 0  # the eval stays dropped afterwards
        node.set_eval(chess.engine.PovScore(chess.engine.Cp(cp), chess.WHITE))
    return game


def test_mine_game_finds_the_win_probability_drop():
    (cand,) = mine_game(annotated_game(-100), set())
    assert cand["ply"] == MIN_PLY
    assert cand["played_uci"] == MOVES[MIN_PLY]
    assert cand["cp_before_white"] == 0
    assert cand["cp_after_white"] == -100
    assert 0.03 <= cand["gap_wp_mined"] <= 0.35
    assert cand["mover_elo"] == 1800  # white moved, so WhiteElo


def test_mine_game_skips_absurd_blunders_and_noise():
    assert mine_game(annotated_game(-5000), set()) == []  # gap too large
    assert mine_game(annotated_game(-5), set()) == []  # gap too small


def test_mine_game_skips_bullet():
    assert mine_game(annotated_game(-100, time_control="60+0"), set()) == []


def test_mine_game_dedupes_against_seen_positions():
    seen: set[str] = set()
    assert len(mine_game(annotated_game(-100), seen)) == 1
    assert mine_game(annotated_game(-100), seen) == []  # same position, second game
