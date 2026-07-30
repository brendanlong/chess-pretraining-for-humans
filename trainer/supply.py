"""Report how much of the bank each user rating can actually be served.

Selection never fails, which is the problem: `server.pick_item` takes the 30
items nearest the target difficulty however far away they are, so a rating with
almost nothing near it is quietly served items from outside the band, and
nothing in the app says so. This is the thing that says so.

Each row is a user rating: the item rating that targets 80% expected score, the
win-probability gap that rating means, how many learnable items sit within one
jitter of it, and how far from target the 30 items selection would draw from
actually sit. Both last columns are read on an untouched bank, so both flatter
it — `pick_item` skips what a user has already answered, so a thin band erodes
into a wide one as they work through it, which is why the count matters where
the drift still looks small.

    uv run python -m trainer.supply [--db data/items.db] [--floor 1000]

`--gaps` prints supply against gap instead, next to what the floor would need
there. That is the form a plan to fill a hole has to be written in, because gap
is what mining and labeling can steer with (`--min-gap-wp` / `--max-gap-wp` on
both), and it is bounded by what the labeler will admit rather than by what the
bank happens to hold: a band nothing has ever been mined for has to appear as a
row, or the report hides exactly the hole it exists to show.
"""

import argparse
from pathlib import Path

from . import label
from .db import DEFAULT_DB, connect
from .rating import (
    _TARGET_OFFSET,
    SELECTION_JITTER,
    USER_MAX,
    USER_MIN,
    _gap_for_difficulty,
    difficulty_rating,
    target_gap,
)

# `pick_item`'s LIMIT: the pool one trial is drawn from.
SELECTION_POOL = 30


def band(conn, target: float) -> int:
    """Learnable items within one jitter of `target`.

    A density around the point trials are aimed at, not a bound on what gets
    served: `target_item_rating` jitters the target by ±`SELECTION_JITTER`
    before `pick_item` sorts by distance from it, so a trial can come from
    outside this — and always does once the band is thin.
    """
    return conn.execute(
        "SELECT COUNT(*) FROM items WHERE learnable = 1 AND ABS(rating - ?) <= ?",
        (target, SELECTION_JITTER),
    ).fetchone()[0]


def pool_drift(conn, target: float) -> float:
    """Mean |rating - target| over the items `pick_item` would choose from."""
    rows = conn.execute(
        "SELECT ABS(rating - ?) AS d FROM items WHERE learnable = 1 ORDER BY d LIMIT ?",
        (target, SELECTION_POOL),
    ).fetchall()
    return sum(row["d"] for row in rows) / len(rows) if rows else float("nan")


def by_user_rating(conn) -> list[dict]:
    """One row per user rating: what the bank holds where that user is aimed."""
    rows = []
    for user in range(int(USER_MIN), int(USER_MAX) + 1, 100):
        target = user + _TARGET_OFFSET
        rows.append(
            {
                "user": user,
                "target": target,
                "gap": target_gap(user),
                "in_band": band(conn, target),
                "drift": pool_drift(conn, target),
            }
        )
    return rows


def band_width_in_gap(gap: float) -> float:
    """How much *fillable* gap one band spans at this gap.

    A band is ±`SELECTION_JITTER` rating points wide however hard the items in
    it are, but the gap-to-rating curve flattens past its knee, so one band at
    the easy end covers several times the gap a band in the middle does and
    needs proportionally fewer items to reach the same depth.

    Past the knee it flattens without limit — a band centred below
    `SELECTION_JITTER` reaches a gap of 3, which no position has and the
    labeler would refuse anyway. So the edges are clipped to the range the
    labeler admits, which turns that into the truth it stands for: at the ends,
    part of a band cannot be filled, so the part that can needs proportionally
    more items.
    """
    difficulty = difficulty_rating(gap)
    lo = max(_gap_for_difficulty(difficulty + SELECTION_JITTER), label.MIN_GAP_WP)
    hi = min(_gap_for_difficulty(max(difficulty - SELECTION_JITTER, 1e-9)), label.MAX_GAP_WP)
    return max(hi - lo, 1e-9)


def by_gap(conn, step: float, floor: int) -> list[dict]:
    """Supply per gap bin, against what a `floor`-deep band needs there.

    `short` is the mining order: it is stated in the units mining filters on,
    and summing it is the size of the labeling job. It is deliberately a
    per-bin figure while a band spans more than one bin, so neighbours cover
    for each other in reality and the total is an over-order — the direction to
    be wrong in, given the marginal item costs a third of a kilobyte.

    Bins run over what the labeler will admit, not over what the bank holds:
    a gap range nobody has mined yet is the emptiest band there is.
    """
    rows = []
    lo = 0.0
    while lo < label.MAX_GAP_WP:
        middle = lo + step / 2
        if label.MIN_GAP_WP <= middle <= label.MAX_GAP_WP:
            count = conn.execute(
                "SELECT COUNT(*) FROM items WHERE learnable = 1 AND gap_wp >= ? AND gap_wp < ?",
                (lo, lo + step),
            ).fetchone()[0]
            need = floor * step / band_width_in_gap(middle)
            rows.append(
                {
                    "lo": lo,
                    "hi": lo + step,
                    "items": count,
                    "need": need,
                    "short": max(0.0, need - count),
                }
            )
        lo = round(lo + step, 4)
    return rows


def table(header: list[str], rows: list[list[str]]) -> str:
    body = "\n".join(f"| {' | '.join(r)} |" for r in rows)
    return f"| {' | '.join(header)} |\n|{' ---: |' * len(header)}\n{body}"


def main() -> None:
    ap = argparse.ArgumentParser(description="What the bank can serve, and where it can't.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--floor", type=int, default=1000, help="the band depth to aim at")
    ap.add_argument("--gaps", action="store_true", help="tabulate by gap instead")
    ap.add_argument("--gap-step", type=float, default=0.02)
    args = ap.parse_args()

    conn = connect(args.db)
    total, learnable = conn.execute("SELECT COUNT(*), SUM(learnable) FROM items").fetchone()
    print(f"{total} items, {learnable or 0} learnable\n")

    if args.gaps:
        rows = by_gap(conn, args.gap_step, args.floor)
        print(
            table(
                ["gap", "items", "for the floor", "short by"],
                [
                    [
                        f"{r['lo']:.2f}-{r['hi']:.2f}",
                        f"{r['items']}",
                        f"{r['need']:.0f}",
                        f"{r['short']:.0f}",
                    ]
                    for r in rows
                ],
            )
        )
        print(f"\n{sum(r['short'] for r in rows):.0f} items short of {args.floor} in every band")
    else:
        rows = by_user_rating(conn)
        print(
            table(
                ["user rating", "target item", "target gap", "in band", "pool drift"],
                [
                    [
                        f"{r['user']}",
                        f"{r['target']:.0f}",
                        f"{r['gap']:.3f}",
                        f"{r['in_band']}{'*' if r['in_band'] < args.floor else ''}",
                        f"{r['drift']:.0f}",
                    ]
                    for r in rows
                ],
            )
        )
        print(f"\n* thinner than {args.floor} items within ±{SELECTION_JITTER}")


if __name__ == "__main__":
    main()
