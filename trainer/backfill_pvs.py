"""Backfill pv_best/pv_distractor for items labeled before PVs were stored.

Runs a depth-DEPTH_DEEP search restricted to each item's two moves and
stores the resulting lines. Idempotent: only touches rows with NULL pvs.

Usage:
    uv run python -m trainer.backfill_pvs [--workers 4]
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine

from .db import DEFAULT_DB, connect
from .label import DEPTH_DEEP, _engines, get_engine, pv_text


def compute_pvs(item: dict) -> tuple[int, str, str] | None:
    engine = get_engine()
    board = chess.Board(item["fen"])
    pvs = []
    for uci in (item["best_uci"], item["distractor_uci"]):
        move = chess.Move.from_uci(uci)
        info = engine.analyse(
            board, chess.engine.Limit(depth=DEPTH_DEEP), root_moves=[move]
        )
        pvs.append(pv_text(info) or uci)
    return item["id"], pvs[0], pvs[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    conn = connect(args.db)
    items = [
        dict(r)
        for r in conn.execute(
            "SELECT id, fen, best_uci, distractor_uci FROM items WHERE pv_best IS NULL"
        )
    ]
    print(f"backfilling PVs for {len(items)} items", file=sys.stderr)

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(compute_pvs, items):
            item_id, pv_best, pv_d = result
            conn.execute(
                "UPDATE items SET pv_best = ?, pv_distractor = ? WHERE id = ?",
                (pv_best, pv_d, item_id),
            )
            conn.commit()
            done += 1
            if done % 200 == 0:
                print(f"{done}/{len(items)}", file=sys.stderr)
    for engine in _engines:
        engine.quit()
    print(f"done: {done} items backfilled", file=sys.stderr)


if __name__ == "__main__":
    main()
