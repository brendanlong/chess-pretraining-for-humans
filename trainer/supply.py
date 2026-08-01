"""Report how much of the bank each user rating can actually be served.

Selection never fails, which is the problem: `server.pick_item` takes the 30
items nearest the target difficulty however far away they are, so a rating with
almost nothing near it is quietly served items from outside the band, and
nothing in the app says so. This is the thing that says so.

Each row is a user rating: the item rating that targets 80% expected score, the
shallow win-probability gap that rating means, how many learnable items sit
within one jitter of it, how much of that band is made of items whose shallow
gap is *negative* — where the surface recommends the wrong move — and how far
from target the 30 items selection would draw from actually sit. The count and
the drift are read on an untouched bank, so both flatter it: `pick_item` skips
what a user has already answered, so a thin band erodes into a wide one as they
work through it, which is why the count matters where the drift still looks
small.

    uv run python -m trainer.supply [--db data/items.db] [--floor 1000]

`--gaps` prints the same bank against the *deep* gap, which is the only thing
mining and labeling can steer with, next to the difficulties each gap bin
actually produced. Difficulty is a function of the shallow gap, so steering the
deep one aims a shotgun rather than a rifle: the two correlate, but a bin
scatters across the scale and the report says where.
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
    target_gap,
)

# Mirrors `server.SELECTION_POOL`, the pool one trial is drawn from — a copy
# rather than an import, because importing the server opens its database
# connection as a side effect.
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


def band_misleading(conn, target: float) -> float:
    """Share of the band whose shallow gap is negative.

    The hardest thing an item can be: a shallow search doesn't merely fail to
    see the difference, it prefers the wrong move. Worth its own column because
    the difficulty number can't distinguish a band reached by narrow-but-honest
    gaps from one reached by misleading ones, and only the second is training
    the reflex to distrust the obvious.
    """
    rows = conn.execute(
        "SELECT COUNT(*), SUM(shallow_gap < 0) FROM items"
        " WHERE learnable = 1 AND shallow_gap IS NOT NULL AND ABS(rating - ?) <= ?",
        (target, SELECTION_JITTER),
    ).fetchone()
    return rows[1] / rows[0] if rows[0] else float("nan")


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
                "misleading": band_misleading(conn, target),
                "drift": pool_drift(conn, target),
            }
        )
    return rows


def by_gap(conn, step: float) -> list[dict]:
    """What each *deep*-gap bin holds, and where on the scale its items land.

    Mining and labeling filter on the deep gap (`--min-gap-wp`/`--max-gap-wp` on
    both) and difficulty is a function of the shallow one, so a bin does not
    map to a band. The two correlate but they are readings of different
    searches, and a bin's items scatter across the scale — which is why this
    reports where they scattered to instead of an order in items. To deepen a
    thin band, find the bin whose quartiles straddle it and mine there; the
    count and the spread together say how much of what you mine will land.

    Bins run over what the labeler will admit, not over what the bank holds: a
    gap range nobody has mined yet is the emptiest band there is, and has to
    appear as a row or the report hides exactly the hole it exists to show.
    """
    rows = []
    lo = 0.0
    while lo < label.MAX_GAP_WP:
        middle = lo + step / 2
        if label.MIN_GAP_WP <= middle <= label.MAX_GAP_WP:
            ratings = [
                r["rating"]
                for r in conn.execute(
                    "SELECT rating FROM items WHERE learnable = 1"
                    "   AND gap_wp >= ? AND gap_wp < ? ORDER BY rating",
                    (lo, lo + step),
                )
            ]

            def at(q: float, ratings=ratings) -> float:
                return ratings[int(q * (len(ratings) - 1))] if ratings else float("nan")

            rows.append(
                {
                    "lo": lo,
                    "hi": lo + step,
                    "items": len(ratings),
                    "p25": at(0.25),
                    "p50": at(0.50),
                    "p75": at(0.75),
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
        rows = by_gap(conn, args.gap_step)
        print(
            table(
                ["deep gap", "items", "difficulty p25", "median", "p75"],
                [
                    [
                        f"{r['lo']:.2f}-{r['hi']:.2f}",
                        f"{r['items']}",
                        f"{r['p25']:.0f}",
                        f"{r['p50']:.0f}",
                        f"{r['p75']:.0f}",
                    ]
                    for r in rows
                ],
            )
        )
        print("\nmine the bin whose quartiles straddle the band you need")
    else:
        rows = by_user_rating(conn)
        print(
            table(
                [
                    "user rating",
                    "target item",
                    "target gap",
                    "in band",
                    "misleading",
                    "pool drift",
                ],
                [
                    [
                        f"{r['user']}",
                        f"{r['target']:.0f}",
                        f"{r['gap']:.3f}",
                        f"{r['in_band']}{'*' if r['in_band'] < args.floor else ''}",
                        f"{r['misleading']:.0%}",
                        f"{r['drift']:.0f}",
                    ]
                    for r in rows
                ],
            )
        )
        print(f"\n* thinner than {args.floor} items within ±{SELECTION_JITTER}")


if __name__ == "__main__":
    main()
