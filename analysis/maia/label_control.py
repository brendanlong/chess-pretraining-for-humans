"""Stockfish's best move for each control position, at the bank's own depth.

The control set is mined with the gap window opened all the way, so it is the
same positions the bank comes from with the "a human blundered here" filter
switched off. All it needs to be comparable is a best move to score Maia
against — no ladder, no learnability, none of the rest of `trainer.label`.

    label_control.py control-candidates.jsonl control.jsonl
"""

import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

import chess
import chess.engine

DEPTH = 18  # trainer.label.DEPTH_DEEP, so the two are the same ground truth
WORKERS = 20

_local = threading.local()


def engine():
    if not hasattr(_local, "engine"):
        _local.engine = chess.engine.SimpleEngine.popen_uci("stockfish")
        _local.engine.configure({"Threads": 1, "Hash": 128})
    return _local.engine


def best(row):
    board = chess.Board(row["fen"])
    info = engine().analyse(board, chess.engine.Limit(depth=DEPTH))
    return {**row, "best_uci": info["pv"][0].uci(), "n_legal": board.legal_moves.count()}


def main():
    with open(sys.argv[1]) as f:
        rows = [json.loads(line) for line in f]
    with ThreadPoolExecutor(WORKERS) as pool, open(sys.argv[2], "w") as out:
        for i, r in enumerate(pool.map(best, rows)):
            out.write(json.dumps(r) + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(rows)}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
