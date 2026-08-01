"""Re-measure the difficulty curve from the bank, and score the alternatives.

`rating.GAP_SLOPE` and `rating.SHALLOW_PLIES` are numbers somebody fitted once.
This is the fitting, so that retuning them is a command rather than an
archaeology project, and so that a claim in a comment can be checked against
the rows it was taken from. CALIBRATION.md is the prose; this is the arithmetic.

    uv run python -m trainer.fit_difficulty            # the fit behind GAP_SLOPE
    uv run python -m trainer.fit_difficulty --windows  # score every window 1..k
    uv run python -m trainer.fit_difficulty --axes     # shallow vs deep vs depth

Pure Python on purpose: this has to run wherever the bank does, and a fit that
needs a scientific stack installed first is a fit nobody re-runs.
"""

import argparse
import random
import sqlite3
from pathlib import Path

from .db import DEFAULT_DB, connect
from .rating import SHALLOW_PLIES

# Errors are binned by the strength of the player who made them; a band too thin
# to have a stable quantile is dropped rather than allowed to swing the fit.
BAND_WIDTH = 100
MIN_BAND = 120
QUANTILE = 0.75


def quantile(sorted_values: list[float], q: float) -> float:
    """Linear interpolation between order statistics — numpy's default, and so
    the published constants'. A nearest-rank quantile is a different estimator
    and moves the fitted slope by several percent, which is enough to make a
    re-measurement look like a disagreement when it is only a convention."""
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


def rows(conn, untargeted: set[str] | None) -> list[tuple[float, list[float], float]]:
    """(mover_elo, ladder, deep gap) for every error the fit is entitled to use.

    'game'-source only, because the whole method rests on the item recording a
    mistake a named human really made; and learnable only, because an item
    nobody is served says nothing about who could see it.
    """
    out = []
    for row in conn.execute(
        "SELECT fen, mover_elo, gap_ladder, gap_wp FROM items"
        " WHERE learnable = 1 AND distractor_source = 'game' AND mover_elo IS NOT NULL"
        "   AND gap_ladder IS NOT NULL AND gap_ladder != ''"
    ):
        if untargeted is not None and row["fen"] not in untargeted:
            continue
        out.append(
            (
                float(row["mover_elo"]),
                [float(x) for x in row["gap_ladder"].split()],
                float(row["gap_wp"]),
            )
        )
    return out


def fit(sample: list[tuple[float, float]]) -> tuple[float, int]:
    """Slope in rating points per unit gap, from (strength, gap) pairs.

    Bin by strength, take a quantile of the gap in each band, and fit strength
    back against it weighted by band size. The direction matters: this asks
    "how big is the error a player of this strength still makes", which is a
    boundary, and not "how strong is the player who makes an error this big",
    which is a mean over everyone who ever blundered. Returns (slope, bands).
    """
    bands: dict[int, list[float]] = {}
    for elo, gap in sample:
        bands.setdefault(int(elo // BAND_WIDTH), []).append(gap)
    cells = [
        (key * BAND_WIDTH + BAND_WIDTH / 2, quantile(sorted(gaps), QUANTILE), len(gaps))
        for key, gaps in bands.items()
        if len(gaps) >= MIN_BAND
    ]
    if len(cells) < 3:
        return float("nan"), len(cells)
    n = sum(w for _, _, w in cells)
    mean_x = sum(x * w for _, x, w in cells) / n
    mean_y = sum(y * w for y, _, w in cells) / n
    var = sum(w * (x - mean_x) ** 2 for _, x, w in cells)
    cov = sum(w * (x - mean_x) * (y - mean_y) for y, x, w in cells)
    if var == 0:
        # Every band's quantile identical: the sample says nothing about slope,
        # and a traceback would be a worse way to say so than a number that
        # propagates and prints as "nan".
        return float("nan"), len(cells)
    return -cov / var, len(cells)


def spread(sample: list[tuple[float, float]], bins: int = 9) -> float:
    """How far the tail of erring strength moves from the easiest bin to the
    hardest, when items are ordered by a candidate difficulty measure.

    The score every candidate axis is compared on. `gap` is passed already
    signed so that larger means harder, so this is directly comparable across
    measures on different scales — which is the only way to ask whether the
    shallow gap beats the deep one without first having a curve for each.
    """
    ordered = sorted(sample, key=lambda pair: pair[1])
    size = len(ordered) // bins
    tails = [
        quantile(sorted(elo for elo, _ in ordered[i * size : (i + 1) * size]), QUANTILE)
        for i in range(bins)
    ]
    return tails[-1] - tails[0]


def bootstrap(sample: list[tuple[float, float]], draws: int, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    slopes = sorted(
        fit([sample[rng.randrange(len(sample))] for _ in range(len(sample))])[0]
        for _ in range(draws)
    )
    return quantile(slopes, 0.025), quantile(slopes, 0.975)


def window(ladder: list[float], plies: int) -> float:
    return sum(ladder[:plies]) / plies if len(ladder) >= plies else ladder[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-measure the difficulty curve.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--untargeted",
        type=Path,
        help="a bank mined without aiming at gap bands; the fit is restricted to "
        "positions it holds, because targeting is selection on the fitted quantity",
    )
    ap.add_argument("--windows", action="store_true", help="score every window 1..k")
    ap.add_argument(
        "--axes", action="store_true", help="score the candidate axes against each other"
    )
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    untargeted = None
    if args.untargeted:
        # Read-only, and not through `connect`: this is a reference bank, often
        # a chmod-444 fixture, and migrating it is neither wanted nor allowed.
        reference = sqlite3.connect(f"file:{args.untargeted}?mode=ro", uri=True)
        untargeted = {fen for (fen,) in reference.execute("SELECT fen FROM items")}
    data = rows(connect(args.db), untargeted)
    print(f"{len(data)} errors{' from the untargeted bank' if untargeted else ''}")
    if untargeted is None:
        print("  (no --untargeted: a bank mined at chosen gap bands will bias this)")

    if args.axes:
        print("\nelo-tail spread, larger is a better difficulty measure:")
        for name, key in (
            (f"shallow gap (1..{SHALLOW_PLIES})", lambda lad, dp: -window(lad, SHALLOW_PLIES)),
            ("gap at one ply", lambda lad, dp: -lad[0]),
            ("gap at full depth", lambda lad, dp: -dp),
        ):
            print(f"  {name:<26} {spread([(elo, key(lad, dp)) for elo, lad, dp in data]):+7.1f}")
        return

    if args.windows:
        print("\nwindow  slope  bands   elo-tail spread")
        for plies in range(1, 19):
            sample = [(elo, window(lad, plies)) for elo, lad, _ in data]
            slope, bands = fit(sample)
            signed = [(elo, -gap) for elo, gap in sample]
            mark = " <- SHALLOW_PLIES" if plies == SHALLOW_PLIES else ""
            print(f"  1..{plies:<2} {slope:7.0f} {bands:>6}   {spread(signed):+7.1f}{mark}")
        return

    sample = [(elo, window(lad, SHALLOW_PLIES)) for elo, lad, _ in data]
    slope, bands = fit(sample)
    print(f"\nGAP_SLOPE over {bands} strength bands: {slope:.0f} rating points per unit gap")
    if args.bootstrap:
        lo, hi = bootstrap(sample, args.bootstrap)
        print(f"  95% CI [{lo:.0f}, {hi:.0f}] over {args.bootstrap} resamples")
    gaps = sorted(gap for _, gap in sample)
    print(f"  gaps the fit saw: {gaps[0]:+.3f}..{gaps[-1]:+.3f}")
    # The check that this is the same method that produced the number published
    # for the deep gap when the deep gap was the axis. If this stops returning
    # something very near it, the estimator has drifted and the shallow figure
    # above is not comparable with anything.
    deep, deep_bands = fit([(elo, dp) for elo, _, dp in data])
    print(f"\nthe same method against `gap_wp` over {deep_bands} bands: {deep:.0f}")
    print("  (was published as 6096 when the deep gap was the axis)")


if __name__ == "__main__":
    main()
