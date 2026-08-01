"""Label mined candidates with Stockfish and build the item bank.

For each candidate position:

1. Deep multipv-2 search finds the position's best move (the correct answer —
   full-strength ground truth, per the design decision that being wrong
   against real truth beats learning to prefer weaker moves).
2. The distractor is the move actually played in the game when it differs
   from the best move (those are the errors humans actually make); otherwise
   the deep search's second choice ('multipv' provenance).
3. Both moves are re-searched to DEPTH_SHALLOW, keeping the evaluation at
   every iteration on the way. The shallowest one from which the two moves
   stay the right way round is how far ahead the position has to be read
   (`solution_depth`); a position that isn't the right way round even at
   DEPTH_SHALLOW is not learnable and is never served (the label is still
   correct — the item is noise).
4. The deep evals are converted to win probability, and the gap together with
   that depth fixes the item's difficulty (small gap, deep read = hard = high
   rating). That mapping is all difficulty is: nothing downstream revises it,
   so an item means the same thing to every user and on every deployment.

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
# The deepest read an item is allowed to demand, and so the top of the
# `solution_depth` scale. Everything from 1 to here is graded onto difficulty
# by `rating.difficulty_rating`; past here the answer isn't reachable from the
# surface at all and the item is dropped. The cap is what makes "how far you
# have to look" a bounded axis rather than an open-ended one — but it is now
# only the *ceiling* on lookahead, not the amount asked of everybody.
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
# One engine per worker, each on one thread: a laptop-sized default that leaves
# the machine usable, and the arrangement that goes fastest anyway. Stockfish
# scales badly across threads compared with running independent searches, so on
# a bigger box raise `--workers` toward the core count rather than `--threads`
# (measured on 24 cores: 20x1 labels 1.6x faster than 8x2). One thread is also
# the only setting `solution_depth` can be measured on, so leaving it here means
# the measurement costs nothing to protect.
ENGINE_WORKERS = 8
ENGINE_THREADS = 1


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


def winprob_ladder(
    engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move
) -> dict[int, float]:
    """What one move is worth at every search depth up to DEPTH_SHALLOW.

    Read off the engine's own iterative deepening rather than by re-searching
    per depth, so the whole ladder costs what the single fixed-depth search it
    replaces cost — the deepest iteration dominates the ones below it.

    The transposition table is emptied first, and that is the whole measurement:
    a shallow search that can look up what a deep search already stored is not a
    shallow search. The deep pass in `label_candidate` runs on this same
    position moments earlier, so without this the engine reports a two-ply read
    of a conclusion it reached at eighteen. That is not a small correction:
    measured against the real pipeline, leaving the table warm changes the depth
    on about a third of positions (49 of 160 sampled) and flips learnable either
    way on about one in sixteen. It costs under a tenth of a millisecond, and it
    is what makes the number mean the same thing here and in
    `trainer.backfill_depth`, where no deep pass precedes it.
    """
    engine.configure({"Clear Hash": None})
    ladder: dict[int, float] = {}
    with engine.analysis(
        board, chess.engine.Limit(depth=DEPTH_SHALLOW), root_moves=[move]
    ) as analysis:
        for info in analysis:
            depth, score = info.get("depth"), info.get("score")
            # Only depths inside the ladder: Stockfish reports `seldepth` beyond
            # the iteration it is on, and finishing an iteration can carry it a
            # step past the limit.
            if score is not None and depth is not None and 1 <= depth <= DEPTH_SHALLOW:
                cp, mate = pov_parts(score, board.turn)
                ladder[depth] = score_to_winprob(cp, mate)
    return ladder


def shallowest_settled(best: dict[int, float], distractor: dict[int, float]) -> int | None:
    """The shallowest depth from which two ladders stay the right way round.

    "And stay" is the point. A position whose ordering flips back and forth was
    never settled at the depth it first happened to look right, and calling that
    depth the answer would say a comparison is easy on the strength of a
    coincidence. So the walk goes down from the deepest rung, not up from the
    first — which also makes it cheap.

    A depth only one of the two searches reported is stepped over rather than
    ending the walk: a rung with nothing to compare against is no evidence
    either way, and reading it as disagreement would inflate every depth above
    it.

    None means no depth settles it, which is the learnability filter: the answer
    isn't reachable from the surface, so the item is noise however correct its
    label is.
    """
    found = None
    for depth in sorted(set(best) & set(distractor), reverse=True):
        if best[depth] <= distractor[depth]:
            break
        found = depth
    return found


def solution_depth(
    engine: chess.engine.SimpleEngine, board: chess.Board, best: chess.Move, distractor: chess.Move
) -> int | None:
    """How far ahead this comparison has to be read, in plies; None if too far.

    Searched on one thread whatever the labeler is otherwise running, and then
    put back. A parallel search splits the tree differently every time it runs,
    which is a fine price for a number that only has to be roughly right — but
    this one *defines* an item, so two runs over the same position have to reach
    the same answer or the bank stops meaning anything. Measured over 60
    positions: one thread agrees with itself 60 times out of 60, two threads 51.
    The ladder is shallow enough that giving up the threads costs nothing, and
    on one thread it is reproducible across engine processes and across the
    order positions are fed in — which is what lets `trainer.backfill_depth`
    reach the same answer as this on a position it never saw labeled. It is one
    Stockfish build that has to agree with itself, not two: a bank labeled
    across an engine upgrade holds depths from both, which is a reason to
    re-measure a bank rather than something this can prevent.

    Guarded because the swap reallocates the thread pool twice, and the default
    is already one thread, so the common path should not pay for a setting it
    isn't using.
    """
    if ENGINE_THREADS != 1:
        engine.configure({"Threads": 1})
    try:
        return shallowest_settled(
            winprob_ladder(engine, board, best), winprob_ladder(engine, board, distractor)
        )
    finally:
        if ENGINE_THREADS != 1:
            engine.configure({"Threads": ENGINE_THREADS})


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

    depth = solution_depth(engine, board, best, distractor)

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
        # 0, not NULL, when no depth settles it: that is a verdict, and a NULL
        # would file it with the rows nobody has measured yet.
        "solution_depth": depth or 0,
        "learnable": int(depth is not None),
        "depth_deep": DEPTH_DEEP,
        "depth_shallow": DEPTH_SHALLOW,
        "rating": difficulty_rating(gap_wp, depth),
        "ply": cand["ply"],
        "game_url": cand["game_url"],
        "mover_elo": cand["mover_elo"],
        "time_control": cand["time_control"],
    }


def main() -> None:
    global MIN_GAP_WP, MAX_GAP_WP, ENGINE_WORKERS, ENGINE_THREADS
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-gap-wp", type=float, default=MIN_GAP_WP)
    parser.add_argument("--max-gap-wp", type=float, default=MAX_GAP_WP)
    parser.add_argument("--workers", type=int, default=ENGINE_WORKERS)
    parser.add_argument("--threads", type=int, default=ENGINE_THREADS)
    args = parser.parse_args()
    MIN_GAP_WP, MAX_GAP_WP = args.min_gap_wp, args.max_gap_wp
    ENGINE_WORKERS, ENGINE_THREADS = args.workers, args.threads

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
                    solution_depth, learnable, depth_deep, depth_shallow, rating,
                    ply, game_url, mover_elo, time_control)
                   VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
                    :cp_best, :mate_best, :cp_distractor, :mate_distractor,
                    :wp_best, :wp_distractor, :gap_wp, :pv_best, :pv_distractor,
                    :solution_depth, :learnable, :depth_deep, :depth_shallow, :rating,
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
