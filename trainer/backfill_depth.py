"""Measure `solution_depth` on items labeled before it existed.

Difficulty is a function of the gap the *shallow* end of the search saw, so a
row with no ladder measured has no difficulty of its own and is being served at
whatever the deep gap alone once said. This runs the ladder over those rows and
fills in the three columns that come off it.

Idempotent, and NULL is what makes it so: a row the ladder has already answered
holds a depth or the 0 that says even the deepest search doesn't settle it, and
neither is NULL. That second answer retires the item, which is rare and is not
a judgement about difficulty — the scale reaches as far as the engine can see,
so nothing is retired for being hard. It means the deep pass and a search
restricted to the two moves disagree about which is better, so the item has no
answer to teach. Responses already given to it stay interpretable either way.

    uv run python -m trainer.backfill_depth [--workers 8]

`items.rating` is not written here: `db.connect` re-derives it from the column
it is a function of, and unlike `trainer.push_items` this runs on the pipeline's
own bank, which no server has open.
"""

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess

from .db import DEFAULT_DB, connect
from .label import _engines, gap_ladder_text, get_engine, measure_lookahead, shallowest_settled
from .rating import shallow_gap_of


def measure(item: dict) -> tuple[int, int | None, str]:
    ladder = measure_lookahead(
        get_engine(),
        chess.Board(item["fen"]),
        chess.Move.from_uci(item["best_uci"]),
        chess.Move.from_uci(item["distractor_uci"]),
    )
    return item["id"], shallowest_settled(ladder), gap_ladder_text(ladder)


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
        for item_id, depth, ladder in pool.map(measure, items):
            conn.execute(
                "UPDATE items SET solution_depth = ?, gap_ladder = ?, shallow_gap = ?,"
                " learnable = ? WHERE id = ?",
                (depth or 0, ladder, shallow_gap_of(ladder), int(depth is not None), item_id),
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
