"""Label mined candidates with Stockfish and build the item bank.

For each candidate position:

1. Deep multipv-2 search finds the position's best move (the correct answer —
   full-strength ground truth, per the design decision that being wrong
   against real truth beats learning to prefer weaker moves).
2. The distractor is the move actually played in the game when it differs
   from the best move (those are the errors humans actually make); otherwise
   the deep search's second choice ('multipv' provenance).
3. Both moves are re-evaluated at shallow depth. If the shallow search
   disagrees with the deep search about which move is better, the item's
   answer hinges on deep calculation rather than surface features, so it is
   marked not learnable and never served (label is correct, item is noise).
4. The deep evals are converted to win probability and the gap fixes the
   item's difficulty (small gap = hard = high rating). That mapping is all
   difficulty is: nothing downstream revises it, so an item means the same
   thing to every user and on every deployment.

Usage:
    uv run python -m trainer.label data/candidates.jsonl [--limit N]
"""

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

from .db import DEFAULT_DB, connect
from .rating import difficulty_rating
from .winprob import score_to_winprob

DEPTH_DEEP = 18
DEPTH_SHALLOW = 8
PV_PLIES = 8  # how much of each line to keep for the reveal replay
MIN_GAP_WP = 0.015
# The easy end has to reach past where beginners are aimed, not stop short of
# it. A user at USER_START is targeting a gap around 0.39 and the floor of the
# user scale around 0.56, so a bank capped much below that serves everyone weak
# the same sliver of items at the boundary — which is the failure `rating`'s
# curve exists to prevent, reintroduced through the labeler instead. This is
# above the widest gap the current bank holds (0.648), so it binds on nothing.
MAX_GAP_WP = 0.70
ENGINE_WORKERS = 8
ENGINE_THREADS = 2


_local = threading.local()
_engines: list[chess.engine.SimpleEngine] = []


def get_engine() -> chess.engine.SimpleEngine:
    if not hasattr(_local, "engine"):
        engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        engine.configure({"Threads": ENGINE_THREADS, "Hash": 128})
        _local.engine = engine
        _engines.append(engine)
    return _local.engine


def pov_parts(score: chess.engine.PovScore, turn: chess.Color) -> tuple[int | None, int | None]:
    pov = score.pov(turn)
    return pov.score(), pov.mate()


def pv_text(info: chess.engine.InfoDict) -> str:
    return " ".join(m.uci() for m in info.get("pv", [])[:PV_PLIES])


def label_candidate(cand: dict) -> dict | None:
    engine = get_engine()
    board = chess.Board(cand["fen"])
    played = chess.Move.from_uci(cand["played_uci"])
    if played not in board.legal_moves or board.legal_moves.count() < 2:
        return None

    deep = engine.analyse(board, chess.engine.Limit(depth=DEPTH_DEEP), multipv=2)
    if len(deep) < 2 or not deep[0].get("pv"):
        return None
    best = deep[0]["pv"][0]
    cp_best, mate_best = pov_parts(deep[0]["score"], board.turn)
    pv_best = pv_text(deep[0])

    if best != played:
        distractor, source = played, "game"
        if deep[1].get("pv") and deep[1]["pv"][0] == played:
            cp_d, mate_d = pov_parts(deep[1]["score"], board.turn)
            pv_d = pv_text(deep[1])
        else:
            info = engine.analyse(board, chess.engine.Limit(depth=DEPTH_DEEP), root_moves=[played])
            cp_d, mate_d = pov_parts(info["score"], board.turn)
            pv_d = pv_text(info) or played.uci()
    else:
        # Deep search says the game move was actually best (the server evals
        # that flagged this position were shallower). Fall back to multipv 2.
        if not deep[1].get("pv"):
            return None
        distractor, source = deep[1]["pv"][0], "multipv"
        cp_d, mate_d = pov_parts(deep[1]["score"], board.turn)
        pv_d = pv_text(deep[1])

    wp_best = score_to_winprob(cp_best, mate_best)
    wp_d = score_to_winprob(cp_d, mate_d)
    # Rounded before anything derives from it, so that the gap the row stores
    # is the gap its difficulty was computed from.
    gap_wp = round(wp_best - wp_d, 4)
    if not (MIN_GAP_WP <= gap_wp <= MAX_GAP_WP):
        return None

    # Learnability filter: does a shallow search see the same ordering?
    shallow_wp = {}
    for move in (best, distractor):
        info = engine.analyse(board, chess.engine.Limit(depth=DEPTH_SHALLOW), root_moves=[move])
        cp_s, mate_s = pov_parts(info["score"], board.turn)
        shallow_wp[move] = score_to_winprob(cp_s, mate_s)
    learnable = int(shallow_wp[best] > shallow_wp[distractor])

    return {
        "fen": cand["fen"],
        "best_uci": best.uci(),
        "distractor_uci": distractor.uci(),
        "distractor_source": source,
        "cp_best": cp_best,
        "mate_best": mate_best,
        "cp_distractor": cp_d,
        "mate_distractor": mate_d,
        "wp_best": round(wp_best, 4),
        "wp_distractor": round(wp_d, 4),
        "gap_wp": gap_wp,
        "pv_best": pv_best,
        "pv_distractor": pv_d,
        "learnable": learnable,
        "depth_deep": DEPTH_DEEP,
        "depth_shallow": DEPTH_SHALLOW,
        "rating": difficulty_rating(gap_wp),
        "ply": cand["ply"],
        "game_url": cand["game_url"],
        "mover_elo": cand["mover_elo"],
        "time_control": cand["time_control"],
    }


def main() -> None:
    global MIN_GAP_WP, MAX_GAP_WP
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-gap-wp", type=float, default=MIN_GAP_WP)
    parser.add_argument("--max-gap-wp", type=float, default=MAX_GAP_WP)
    args = parser.parse_args()
    MIN_GAP_WP, MAX_GAP_WP = args.min_gap_wp, args.max_gap_wp

    conn = connect(args.db)
    existing = {row["fen"] for row in conn.execute("SELECT fen FROM items")}

    candidates = []
    with open(args.candidates) as f:
        for line in f:
            cand = json.loads(line)
            if cand["fen"] not in existing:
                candidates.append(cand)
    if args.limit:
        candidates = candidates[: args.limit]
    print(f"labeling {len(candidates)} candidates", file=sys.stderr)

    inserted = 0
    learnable_count = 0
    with ThreadPoolExecutor(max_workers=ENGINE_WORKERS) as pool:
        for i, item in enumerate(pool.map(label_candidate, candidates)):
            if i % 100 == 0:
                print(
                    f"{i}/{len(candidates)} inserted={inserted} learnable={learnable_count}",
                    file=sys.stderr,
                )
            if item is None:
                continue
            conn.execute(
                """INSERT OR IGNORE INTO items
                   (fen, best_uci, distractor_uci, distractor_source,
                    cp_best, mate_best, cp_distractor, mate_distractor,
                    wp_best, wp_distractor, gap_wp, pv_best, pv_distractor,
                    learnable, depth_deep, depth_shallow, rating,
                    ply, game_url, mover_elo, time_control)
                   VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
                    :cp_best, :mate_best, :cp_distractor, :mate_distractor,
                    :wp_best, :wp_distractor, :gap_wp, :pv_best, :pv_distractor,
                    :learnable, :depth_deep, :depth_shallow, :rating,
                    :ply, :game_url, :mover_elo, :time_control)""",
                item,
            )
            conn.commit()  # commit per item: a trainer server may share the db
            inserted += 1
            learnable_count += item["learnable"]
    conn.commit()
    for engine in _engines:
        engine.quit()  # otherwise their non-daemon threads keep the process alive
    print(f"done: inserted={inserted} learnable={learnable_count}", file=sys.stderr)


if __name__ == "__main__":
    main()
