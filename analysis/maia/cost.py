"""When Maia doesn't pick Stockfish's move, how much is that worth?

Top-1 agreement is a harsh metric and on its own an unreadable one: a position
holding three moves inside a hundredth of a win probability scores a miss for
picking any but one of them, and so does a position where Maia hangs a rook.
Those are not the same disagreement. This prices Maia's own choice against the
best move, on the same deep search the bank is labeled with, so the rate can be
read next to the severity.

Takes any file `policy.py` ran over — the bank or a control set — and the
policy dump beside it:

    cost.py items.jsonl maia2.jsonl cost-items.jsonl [--sample 2500]

`--sample` because the bank has more disagreements than are worth a deep search
each, and the quantiles it prints settle long before that.
"""

import argparse
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import numpy as np
from agreement import read

# The one place the probe reaches into the app. Borrowing the conversion rather
# than restating it is the point: a cost in different win-probability units from
# the bank's own would not be comparable with anything in CALIBRATION.md.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from trainer.winprob import cp_to_winprob

DEPTH = 18  # trainer.label.DEPTH_DEEP, so this is the bank's own ground truth
WORKERS = 20
_local = threading.local()
# Every engine handed out, so they can all be shut down at the end. A
# `SimpleEngine` owns a subprocess and a thread, and neither is a daemon: one
# left open holds the interpreter after the last result is printed, so a script
# that skips this appears to hang having already done its work.
_engines = []
_engines_lock = threading.Lock()


def engine():
    if not hasattr(_local, "engine"):
        _local.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        _local.engine.configure({"Threads": 1, "Hash": 128})
        with _engines_lock:
            _engines.append(_local.engine)
    return _local.engine


def close_engines():
    with _engines_lock:
        while _engines:
            _engines.pop().quit()


def winprob(board, uci):
    # Cleared between the two moves of a position as well as between positions.
    # Both searches here are restricted to one root move, so the second would
    # otherwise read the first's table — the two numbers are subtracted from
    # each other, which makes that a systematic bias rather than noise, and it
    # is the same trap `trainer.label` clears the hash for. It also makes the
    # result independent of which positions a worker happened to see first,
    # without which two runs of this disagree by about a point.
    engine().configure({"Clear Hash": None})
    info = engine().analyse(
        board, chess.engine.Limit(depth=DEPTH), root_moves=[chess.Move.from_uci(uci)]
    )
    score = info["score"].pov(board.turn)
    if score.mate() is not None:
        return 1.0 if score.mate() > 0 else 0.0
    return cp_to_winprob(score.score())


def price(row):
    board = chess.Board(row["fen"])
    return {
        "id": row["id"],
        "wp_best": winprob(board, row["best_uci"]),
        "wp_maia": winprob(board, row["maia_top"]),
        "gap_wp": row.get("gap_wp"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("policy")
    ap.add_argument("dst")
    ap.add_argument("--sample", type=int, default=2500)
    ap.add_argument("--elo", type=int, default=1500)
    args = ap.parse_args()

    items = {r.get("id", i): r for i, r in enumerate(read(args.items))}
    pol = {r["id"]: r for r in read(args.policy) if r["elo"] == args.elo}
    disagree = [
        {**items[i], "id": i, "maia_top": pol[i]["top"]}
        for i in items
        if pol[i]["top"] != items[i]["best_uci"]
    ]
    print(
        f"maia-{args.elo} disagrees on {len(disagree)}/{len(items)} "
        f"({len(disagree) / len(items):.1%})",
        file=sys.stderr,
    )
    if len(disagree) > args.sample:
        random.seed(11)
        disagree = random.sample(disagree, args.sample)

    priced = []
    try:
        with ThreadPoolExecutor(WORKERS) as pool, open(args.dst, "w") as out:
            for r in pool.map(price, disagree):
                out.write(json.dumps(r) + "\n")
                priced.append(r)
    finally:
        close_engines()

    # Clipped at zero: a deep search restricted to one move can land a hair
    # above the unrestricted one, which is search noise and not Maia finding
    # something better than best.
    gap = np.clip(np.array([r["wp_best"] - r["wp_maia"] for r in priced]), 0, None)
    print(f"\nwin probability maia-{args.elo}'s move gives up, over {len(gap)} disagreements\n")
    for q in (0.5, 0.75, 0.9):
        print(f"  p{int(q * 100):<4} {np.quantile(gap, q):.4f}")
    print(f"  mean  {gap.mean():.4f}\n")
    # 0.03 is `trainer.mine.MIN_GAP_WP`: below it this project does not consider
    # the move an error at all, which is the only non-arbitrary line available.
    print(f"  under mining's error floor (<0.03)  {np.mean(gap < 0.03):>6.1%}")
    print(f"  under 0.10                          {np.mean(gap < 0.10):>6.1%}")
    print(f"  a real blunder (>=0.25)             {np.mean(gap >= 0.25):>6.1%}")
    own = [r["gap_wp"] for r in priced]
    if all(g is not None for g in own):
        beats = np.mean(gap < np.array(own))
        print(f"\n  better than the move the human played  {beats:>6.1%}")


if __name__ == "__main__":
    main()
