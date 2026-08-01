"""Mine candidate discrimination items from a Lichess PGN stream.

Reads PGN from stdin (intended use: stream the head of a monthly Lichess dump
through zstdcat) and emits one JSON candidate per line. A candidate is a real
decision point from a real game: a position where the player made a move that
Lichess's server analysis says was meaningfully worse than best play.

The played move becomes the distractor later; here we only need positions
where (a) server evals exist, (b) the played move lost a calibrated amount of
win probability, and (c) the position wasn't already decided. Precise labels
(the best move, deep/shallow evals, learnability) come from the local
Stockfish pass in label.py.

Usage:
    curl -s -r 0-100000000 <dump-url> | zstdcat 2>/dev/null | \
        uv run python -m trainer.mine --max-candidates 2000 > data/candidates.jsonl
"""

import argparse
import io
import json
import re
import sys
from collections.abc import Iterable, Iterator

import chess
import chess.pgn

from .winprob import cp_to_winprob

# A PGN header is attacker-controlled data the moment the PGN isn't a Lichess
# dump, and this one is the only mined field the app renders as a link. The
# reveal builds that link as a node rather than markup, so this is the second
# of two locks; keeping the bank clean is worth the four lines regardless.
GAME_URL_RE = re.compile(r"^https://lichess\.org/[A-Za-z0-9]{8,12}$")

MIN_PLY = 12  # skip opening-book territory
MAX_PLY = 90
MAX_ABS_EVAL_CP = 500  # position not already decided (white POV)
MIN_GAP_WP = 0.03  # played move must lose at least this much win prob...
# ...but not be an absurd blunder nobody would consider. This window is read off
# the server's eval, and it pins the *deep* gap tightly — r = 0.95 over 2,431
# labeled candidates, deep minus server centred on 0.000, sd 0.019.
#
# What it does not pin is difficulty, which is a function of the gap a *shallow*
# search saw (`rating.GAP_SLOPE`). Deep and shallow correlate at 0.79 and no
# better, so a 0.10-wide window here arrives spread over roughly a 0.25-wide
# range of shallow gaps: aiming it is a shotgun, not a rifle. Measured over the
# 24,989-item bank, a window lands about 7% of what it labels in any one
# difficulty band, and about half of it somewhere in the ten bands that make up
# the easy end. Filling a thin band is therefore still affordable — order twice
# what you need and mine wide — but the arithmetic is per-region, not per-band,
# and `trainer.supply --gaps` is what prints where a window actually went.
#
# What the window does not do is *bound* difficulty — `label.MAX_GAP_WP` is the
# one that binds, and the bank reaches to a 0.70 gap because its easy end was
# mined with this raised on the command line, which is what the flags are for.
MAX_GAP_WP = 0.35
# What the two above mean when nobody has overridden them, so a candidate can
# say whether it came through the full window or a chosen slice of it.
DEFAULT_MIN_GAP_WP, DEFAULT_MAX_GAP_WP = MIN_GAP_WP, MAX_GAP_WP
MIN_BASE_TIME_S = 180  # blitz and slower; bullet errors are mostly mouse slips
MAX_PER_GAME = 2
MIN_PLY_SPACING = 10  # candidates from one game must be far apart


def game_url(site: str) -> str:
    """The game's URL, or "" if it isn't one we'd put in front of a user."""
    site = site.strip()
    return site if GAME_URL_RE.match(site) else ""


def raw_games(stream: Iterable[str]) -> Iterator[str]:
    """Split a PGN stream into per-game text without parsing moves.

    A PGN game is a header section and a movetext section, each followed by a
    blank line. Splitting on that structure lets us substring-test for %eval
    before paying for a full parse (only ~6% of Lichess games are analyzed).
    """
    chunks: list[str] = []
    blank_seen = 0
    for line in stream:
        chunks.append(line)
        if line.strip() == "":
            blank_seen += 1
            if blank_seen == 2:
                yield "".join(chunks)
                chunks = []
                blank_seen = 0


def score_cp(pov_score) -> int | None:
    """White-POV cp, or None for mate scores (skipped in mining)."""
    if pov_score.mate() is not None:
        return None
    return pov_score.score()


def base_time_seconds(headers: chess.pgn.Headers) -> int:
    tc = headers.get("TimeControl", "-")
    try:
        return int(tc.split("+")[0])
    except ValueError:
        return 0


def mine_game(game: chess.pgn.Game, seen_fens: set[str]) -> list[dict]:
    headers = game.headers
    if base_time_seconds(headers) < MIN_BASE_TIME_S:
        return []

    candidates: list[dict] = []
    last_candidate_ply = -MIN_PLY_SPACING
    prev_eval = None  # PovScore after the previous move
    board = game.board()

    for node in game.mainline():
        move = node.move
        this_eval = node.eval()
        eval_before, prev_eval = prev_eval, this_eval
        ply = board.ply()
        mover = board.turn
        position = board.copy(stack=False)
        board.push(move)

        if len(candidates) >= MAX_PER_GAME:
            break
        if eval_before is None or this_eval is None:
            continue
        if not (MIN_PLY <= ply <= MAX_PLY):
            continue
        if ply - last_candidate_ply < MIN_PLY_SPACING:
            continue

        cp_before_white = score_cp(eval_before.white())
        cp_after_white = score_cp(this_eval.white())
        if cp_before_white is None or cp_after_white is None:
            continue
        if abs(cp_before_white) > MAX_ABS_EVAL_CP:
            continue

        # Win probability lost by the played move, from the mover's POV.
        sign = 1 if mover == chess.WHITE else -1
        wp_before = cp_to_winprob(sign * cp_before_white)
        wp_after = cp_to_winprob(sign * cp_after_white)
        gap_wp = wp_before - wp_after
        if not (MIN_GAP_WP <= gap_wp <= MAX_GAP_WP):
            continue

        epd = position.epd()
        if epd in seen_fens:
            continue
        seen_fens.add(epd)
        last_candidate_ply = ply

        elo_key = "WhiteElo" if mover == chess.WHITE else "BlackElo"
        candidates.append(
            {
                "fen": position.fen(),
                "played_uci": move.uci(),
                "cp_before_white": cp_before_white,
                "cp_after_white": cp_after_white,
                "gap_wp_mined": round(gap_wp, 4),
                # Whether this run narrowed the window. `rating.GAP_SLOPE` may
                # only be fitted on candidates that didn't, because narrowing it
                # is selection on the very quantity the fit regresses — see
                # CALIBRATION.md. Recorded per candidate because it is a fact
                # about how the row got here, and the alternative is recovering
                # it later by diffing against a bank somebody kept.
                "mined_untargeted": int(
                    (MIN_GAP_WP, MAX_GAP_WP) == (DEFAULT_MIN_GAP_WP, DEFAULT_MAX_GAP_WP)
                ),
                "ply": ply,
                "game_url": game_url(headers.get("Site", "")),
                "mover_elo": int(headers.get(elo_key, 0) or 0),
                "time_control": headers.get("TimeControl", ""),
            }
        )
    return candidates


def main() -> None:
    global MIN_GAP_WP, MAX_GAP_WP
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-candidates", type=int, default=2000)
    parser.add_argument("--min-gap-wp", type=float, default=MIN_GAP_WP)
    parser.add_argument("--max-gap-wp", type=float, default=MAX_GAP_WP)
    args = parser.parse_args()
    MIN_GAP_WP, MAX_GAP_WP = args.min_gap_wp, args.max_gap_wp

    seen_fens: set[str] = set()
    emitted = 0
    games_read = 0
    games_with_evals = 0

    for game_text in raw_games(sys.stdin):
        if emitted >= args.max_candidates:
            break
        games_read += 1
        if games_read % 20000 == 0:
            print(
                f"games={games_read} with_evals={games_with_evals} candidates={emitted}",
                file=sys.stderr,
            )
        if "%eval" not in game_text:
            continue
        games_with_evals += 1
        try:
            game = chess.pgn.read_game(io.StringIO(game_text))
            if game is None:
                continue
            candidates = mine_game(game, seen_fens)
        except Exception:
            continue  # malformed / truncated tail
        for c in candidates:
            print(json.dumps(c))
            emitted += 1

    print(
        f"done: games={games_read} with_evals={games_with_evals} candidates={emitted}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
