"""Measure `solution_depth` on items labeled before it existed.

Difficulty is a function of the gap *and* of how far ahead the position has to
be read, so a row with no depth measured is being served as though it needed
none. This runs `label.solution_depth` over those rows and fills it in.

Idempotent: only rows with a NULL `solution_depth` are touched. It can also
retire one — a position no search up to `label.DEPTH_SHALLOW` gets the right
way round is unlearnable and stops being served, which is a verdict on the
item's noise, not on its answer, so responses already given to it stay
interpretable.

    uv run python -m trainer.backfill_depth [--workers 8]

`items.rating` is not written here: `db.connect` re-derives it from the two
columns it is a function of, so the next open picks the change up.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

from .db import DEFAULT_DB, connect
from .label import _engines, get_engine, solution_depth


def measure(item: dict) -> tuple[int, int | None]:
    board = chess.Board(item["fen"])
    depth = solution_depth(
        get_engine(),
        board,
        chess.Move.from_uci(item["best_uci"]),
        chess.Move.from_uci(item["distractor_uci"]),
    )
    return item["id"], depth


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    conn = connect(args.db)
    items = [
        dict(r)
        for r in conn.execute(
            "SELECT id, fen, best_uci, distractor_uci FROM items WHERE solution_depth IS NULL"
        )
    ]
    print(f"measuring lookahead depth for {len(items)} items", file=sys.stderr)

    done = retired = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for item_id, depth in pool.map(measure, items):
            conn.execute(
                "UPDATE items SET solution_depth = ?, learnable = ? WHERE id = ?",
                (depth, int(depth is not None), item_id),
            )
            conn.commit()  # commit per item: a trainer server may share the db
            done += 1
            retired += depth is None
            if done % 500 == 0:
                print(f"{done}/{len(items)} retired={retired}", file=sys.stderr)
    for engine in _engines:
        engine.quit()  # otherwise their non-daemon threads keep the process alive
    print(f"done: {done} measured, {retired} no longer learnable", file=sys.stderr)


if __name__ == "__main__":
    main()
