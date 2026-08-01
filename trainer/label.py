"""Label mined candidates with Stockfish and build the item bank.

For each candidate position:

1. Deep multipv-2 search finds the position's best move (the correct answer —
   full-strength ground truth, per the design decision that being wrong
   against real truth beats learning to prefer weaker moves).
2. The distractor is the move actually played in the game when it differs
   from the best move (those are the errors humans actually make); otherwise
   the deep search's second choice ('multipv' provenance).
3. A second search, restricted to just those two moves and started from an
   empty hash, ranks the pair at every depth on its way to DEPTH_DEEP. The
   whole curve is kept (`gap_ladder`); the shallowest depth from which it
   stays the right way round is how far ahead the position has to be read
   (`solution_depth`). There is no shallower cutoff and nothing is dropped
   for being deep: a comparison that only settles at seventeen plies is an
   extremely hard item, not a rejected one.
4. The mean of the ladder's shallow end (`shallow_gap`) fixes the item's
   difficulty: a narrow gap early is hard, a negative one — where the surface
   recommends the losing move — is harder still. The deep evals are kept for
   the reveal, and say what the answer is worth, but no longer say how hard it
   is. That mapping is all difficulty is: nothing downstream revises it, so an
   item means the same thing to every user and on every deployment. The whole
   ladder is stored rather than only its summary, because the search is the
   expensive half and every reading of it is a guess that will be revised.

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
from .rating import difficulty_rating, shallow_gap_of
from .winprob import score_to_winprob

# The search that decides which move is better, and — because the ladder is read
# off a search to this same depth — the top of the ladder too. There is no
# second, shallower cutoff: an item that only settles at seventeen plies is not
# thrown away for it. Its shallow gap will be narrow or negative, which is what
# rates it hard and what keeps it away from anyone who couldn't have seen it. A
# cutoff would be the thing the measurement exists to replace.
DEPTH_DEEP = 18
PV_PLIES = 8  # how much of each line to keep for the reveal replay
MIN_GAP_WP = 0.015
# A bound on the *deep* gap, which is no longer the axis difficulty is measured
# on — so it no longer bounds difficulty either, and the two are only loosely
# related (they correlate at 0.79). What it still does is refuse positions so
# lopsided that nobody would consider the played move. It is also the lever that
# reaches the easy end of the difficulty scale, because a wide deep gap is the
# best predictor of a wide shallow one the pipeline can filter on: the bank
# holds 28 items at this cap, so raising it is what a refill run there does.
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


def pair_ladder(
    engine: chess.engine.SimpleEngine, board: chess.Board, best: chess.Move, distractor: chess.Move
) -> dict[int, tuple[float, float]]:
    """What the two moves are worth to each other at every depth up to DEPTH_DEEP.

    One multipv-2 search restricted to exactly these two moves, read off the
    engine's own iterative deepening. So the whole ladder costs about one deep
    search rather than eighteen, and — the reason to prefer it to a search per
    move — every rung is the engine *ranking the pair*, which is the same
    question the deep pass answers when it decides which move is better. Two
    isolated searches would instead compare numbers from two alpha-beta windows
    that never saw each other, and they disagree: on a fifth of positions the
    isolated form calls a comparison settled at one ply that a search ranking
    both moves does not settle until eight.

    The transposition table is emptied first, and that is the whole measurement:
    a shallow search that can look up what a deep search already stored is not a
    shallow search. The deep pass in `label_candidate` runs on this same position
    moments earlier, so without this the engine reports a two-ply read of a
    conclusion it reached at eighteen — which changes the depth on about a third
    of positions. It costs under a tenth of a millisecond.
    """
    engine.configure({"Clear Hash": None})
    per_depth: dict[int, dict[chess.Move, float]] = {}
    with engine.analysis(
        board, chess.engine.Limit(depth=DEPTH_DEEP), root_moves=[best, distractor], multipv=2
    ) as analysis:
        for info in analysis:
            depth, score, pv = info.get("depth"), info.get("score"), info.get("pv")
            # `pv` names which of the two this line is about; multipv rank can't,
            # since which move holds rank 1 is the thing being measured. Depths
            # outside the ladder are dropped: Stockfish reports `seldepth` past
            # the iteration it is on, and finishing one can carry it a step past
            # the limit.
            if score is None or not pv or depth is None or not 1 <= depth <= DEPTH_DEEP:
                continue
            cp, mate = pov_parts(score, board.turn)
            per_depth.setdefault(depth, {})[pv[0]] = score_to_winprob(cp, mate)
    return {
        depth: (moves[best], moves[distractor])
        for depth, moves in per_depth.items()
        if best in moves and distractor in moves
    }


def shallowest_settled(ladder: dict[int, tuple[float, float]]) -> int | None:
    """The shallowest depth from which the pair stays the right way round.

    "And stays" is the point. A position whose ordering flips back and forth was
    never settled at the depth it first happened to look right, and calling that
    depth the answer would say a comparison is easy on the strength of a
    coincidence. So the walk goes down from the deepest rung, not up from the
    first — which also makes it cheap.

    A depth the search reported for only one of the two moves never reaches
    here: a rung with nothing to compare against is no evidence either way, and
    reading it as disagreement would inflate every depth above it.

    None means not even DEPTH_DEEP settles it. That is no longer a statement
    about how hard the item is — the scale has room for the hardest thing the
    engine can still see — but about the label itself: a search to the depth
    that picked the best move, restricted to the pair, disagreeing with the pick
    is an item whose answer the engine does not hold steady, so there is nothing
    to teach.
    """
    found = None
    for depth in sorted(ladder, reverse=True):
        wp_best, wp_distractor = ladder[depth]
        if wp_best <= wp_distractor:
            break
        found = depth
    return found


def gap_ladder_text(ladder: dict[int, tuple[float, float]]) -> str:
    """The ladder as it is stored: one gap per depth, shallowest first.

    Gaps rather than the pair of win probabilities, because the gap is the
    comparison and the two evaluations separately are a fact about the position
    that `wp_best`/`wp_distractor` already record at full depth. Rounded to four
    places like `gap_wp`, so a stored number is one a difficulty function can be
    a pure function of.

    Depth is the position in the list, so the rungs have to be contiguous from
    1; a search that skipped one would otherwise silently renumber the rest.
    Stockfish doesn't skip, and a ladder that did is dropped rather than
    guessed at.
    """
    depths = sorted(ladder)
    if depths != list(range(1, len(depths) + 1)):
        return ""
    return " ".join(f"{ladder[d][0] - ladder[d][1]:.4f}" for d in depths)


def measure_lookahead(
    engine: chess.engine.SimpleEngine, board: chess.Board, best: chess.Move, distractor: chess.Move
) -> dict[int, tuple[float, float]]:
    """The lookahead ladder for one item, measured reproducibly.

    Searched on one thread whatever the labeler is otherwise running, and then
    put back. A parallel search splits the tree differently every time it runs,
    which is a fine price for a number that only has to be roughly right — but
    this one *defines* an item, so two runs over the same position have to reach
    the same answer or the bank stops meaning anything. On one thread it is
    reproducible across engine processes, across a preceding deep pass, and
    across the order positions are fed in — which is what lets
    `trainer.backfill_depth` reach the same answer as this on a position it
    never saw labeled. It is one Stockfish build that has to agree with itself,
    not two: a bank labeled across an engine upgrade holds depths from both,
    which is a reason to re-measure a bank rather than something this can
    prevent.

    Guarded because the swap reallocates the thread pool twice, and the default
    is already one thread, so the common path should not pay for a setting it
    isn't using.
    """
    if ENGINE_THREADS != 1:
        engine.configure({"Threads": 1})
    try:
        return pair_ladder(engine, board, best, distractor)
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

    ladder = measure_lookahead(engine, board, best, distractor)
    depth = shallowest_settled(ladder)
    ladder_text = gap_ladder_text(ladder)
    shallow_gap = shallow_gap_of(ladder_text)
    if shallow_gap is None:
        # No usable ladder, so no difficulty. Rare enough to drop rather than
        # invent a number for: an item the scale can't place is an item
        # selection can't aim at.
        return None

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
        "gap_ladder": ladder_text,
        "shallow_gap": shallow_gap,
        "learnable": int(depth is not None),
        "depth_deep": DEPTH_DEEP,
        "rating": difficulty_rating(shallow_gap),
        "ply": cand["ply"],
        "mined_untargeted": cand.get("mined_untargeted"),
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
                    solution_depth, gap_ladder, shallow_gap, learnable, depth_deep, rating,
                    ply, mined_untargeted, game_url, mover_elo, time_control)
                   VALUES (:fen, :best_uci, :distractor_uci, :distractor_source,
                    :cp_best, :mate_best, :cp_distractor, :mate_distractor,
                    :wp_best, :wp_distractor, :gap_wp, :pv_best, :pv_distractor,
                    :solution_depth, :gap_ladder, :shallow_gap, :learnable, :depth_deep, :rating,
                    :ply, :mined_untargeted, :game_url, :mover_elo, :time_control)""",
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
