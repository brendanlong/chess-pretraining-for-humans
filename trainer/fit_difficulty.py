"""Re-measure the difficulty curve from the bank, and score the alternatives.

`rating.GAP_SLOPE` and `rating.SHALLOW_PLIES` are numbers somebody fitted once.
This is the fitting, so that retuning them is a command rather than an
archaeology project, and so that a claim in a comment can be checked against
the rows it was taken from. CALIBRATION.md is the prose; this is the arithmetic.

    uv run python -m trainer.fit_difficulty            # the fit behind GAP_SLOPE
    uv run python -m trainer.fit_difficulty --windows  # score every window 1..k
    uv run python -m trainer.fit_difficulty --axes     # shallow vs deep vs depth

numpy is a dev-group dependency and is not in the deployment; nothing the
server does imports this. Using it rather than hand-rolling the statistics is
deliberate: the published constants were fitted with `numpy.percentile`, and a
quantile that interpolates differently moves the slope by percent — enough to
read as a disagreement with the comment when it is only a convention.
"""

import argparse
from pathlib import Path

import numpy as np

from .db import DEFAULT_DB, connect
from .rating import SHALLOW_PLIES

# Errors are binned by the strength of the player who made them; a band too thin
# to have a stable quantile is dropped rather than allowed to swing the fit.
BAND_WIDTH = 100
MIN_BAND = 120
# A fraction, so every reader of it wants `np.quantile` and not
# `np.percentile` — which takes 0..100 and silently answers a different
# question, one that still fits a plausible-looking slope.
QUANTILE = 0.75


def rows(conn, everything: bool) -> list[tuple[float, list[float], float]]:
    """(mover_elo, ladder, deep gap) for every error the fit is entitled to use.

    'game'-source only, because the whole method rests on the item recording a
    mistake a named human really made; learnable only, because an item nobody is
    served says nothing about who could see it; and `mined_untargeted` only,
    because a gap window aimed at a band is selection on the very quantity being
    regressed. `everything` lifts that last one, which is how you see what the
    bias is worth rather than taking it on faith.
    """
    return [
        (
            float(row["mover_elo"]),
            [float(x) for x in row["gap_ladder"].split()],
            float(row["gap_wp"]),
        )
        for row in conn.execute(
            "SELECT mover_elo, gap_ladder, gap_wp FROM items"
            " WHERE learnable = 1 AND distractor_source = 'game' AND mover_elo IS NOT NULL"
            "   AND gap_ladder != ''" + ("" if everything else " AND mined_untargeted = 1")
        )
    ]


def fit(elo: np.ndarray, gap: np.ndarray) -> tuple[float, int]:
    """Slope in rating points per unit gap. Returns (slope, bands used).

    Bin by strength, take a quantile of the gap in each band, and fit strength
    back against it weighted by band size. The direction matters: this asks
    "how big is the error a player of this strength still makes", which is a
    boundary, and not "how strong is the player who makes an error this big",
    which is a mean over everyone who ever blundered.

    nan when the sample can't say — fewer than three bands, or every band's
    quantile identical. A number that propagates and prints is a better way to
    report that than a traceback.
    """
    cells = []
    for key in np.unique(elo // BAND_WIDTH):
        gaps = gap[elo // BAND_WIDTH == key]
        if gaps.size >= MIN_BAND:
            cells.append(
                (key * BAND_WIDTH + BAND_WIDTH / 2, np.quantile(gaps, QUANTILE), gaps.size)
            )
    if len(cells) < 3:
        return float("nan"), len(cells)
    strength, gap75, weight = np.array(cells).T
    centred = gap75 - np.average(gap75, weights=weight)
    variance = np.sum(weight * centred**2)
    if variance == 0:
        return float("nan"), len(cells)
    covariance = np.sum(weight * centred * (strength - np.average(strength, weights=weight)))
    return -covariance / variance, len(cells)


def spread(elo: np.ndarray, hardness: np.ndarray, bins: int = 9) -> float:
    """How far the tail of erring strength moves from the easiest bin of items
    to the hardest, when they are ordered by a candidate difficulty measure.

    The score every candidate axis is compared on. `hardness` is signed so that
    larger means harder, so this is comparable across measures on different
    scales — which is the only way to ask whether the shallow gap beats the deep
    one without first having a fitted curve for each.
    """
    by_hardness = elo[np.argsort(hardness)]
    tails = [np.quantile(chunk, QUANTILE) for chunk in np.array_split(by_hardness, bins)]
    return float(tails[-1] - tails[0])


def bootstrap(elo: np.ndarray, gap: np.ndarray, draws: int, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    slopes = [
        fit(*(a[i] for a in (elo, gap)))[0]
        for i in (rng.integers(0, elo.size, elo.size) for _ in range(draws))
    ]
    return tuple(np.quantile(slopes, [0.025, 0.975]))


def window(ladder: list[float], plies: int) -> float:
    """The mean of a ladder's first `plies` rungs, or its last if it is shorter.

    Short ladders never reach the fit — `rows` filters them — but averaging a
    different number of rungs for different items would be a different measure
    wearing one name, so the fallback is stated rather than left to slicing.
    """
    return sum(ladder[:plies]) / plies if len(ladder) >= plies else ladder[-1]


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-measure the difficulty curve.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument(
        "--everything",
        action="store_true",
        help="include positions mined at a chosen gap band. Not a measurement — "
        "it is how you check what that selection is worth",
    )
    ap.add_argument("--windows", action="store_true", help="score every window 1..k")
    ap.add_argument(
        "--axes", action="store_true", help="score the candidate axes against each other"
    )
    ap.add_argument("--bootstrap", type=int, default=400)
    args = ap.parse_args()

    data = rows(connect(args.db), args.everything)
    print(f"{len(data)} errors{', including gap-targeted ones' if args.everything else ''}")
    if args.everything:
        print("  (not a measurement — targeting selects on the fitted quantity)")

    elo = np.array([e for e, _, _ in data])
    deep_gap = np.array([d for _, _, d in data])
    shallow = np.array([window(lad, SHALLOW_PLIES) for _, lad, _ in data])

    if args.axes:
        print("\nelo-tail spread, larger is a better difficulty measure:")
        for name, gap in (
            (f"shallow gap (1..{SHALLOW_PLIES})", shallow),
            ("gap at one ply", np.array([lad[0] for _, lad, _ in data])),
            ("gap at full depth", deep_gap),
        ):
            print(f"  {name:<26} {spread(elo, -gap):+7.1f}")
        return

    if args.windows:
        print("\nwindow  slope  bands   elo-tail spread")
        for plies in range(1, 19):
            gap = np.array([window(lad, plies) for _, lad, _ in data])
            slope, bands = fit(elo, gap)
            mark = " <- SHALLOW_PLIES" if plies == SHALLOW_PLIES else ""
            print(f"  1..{plies:<2} {slope:7.0f} {bands:>6}   {spread(elo, -gap):+7.1f}{mark}")
        return

    slope, bands = fit(elo, shallow)
    print(f"\nGAP_SLOPE over {bands} strength bands: {slope:.0f} rating points per unit gap")
    if args.bootstrap:
        lo, hi = bootstrap(elo, shallow, args.bootstrap)
        print(f"  95% CI [{lo:.0f}, {hi:.0f}] over {args.bootstrap} resamples")
    print(f"  gaps the fit saw: {shallow.min():+.3f}..{shallow.max():+.3f}")
    # The check that this is the same method that produced the number published
    # for the deep gap when the deep gap was the axis. If this stops returning
    # something very near it, the estimator has drifted and the shallow figure
    # above is not comparable with anything.
    deep, deep_bands = fit(elo, deep_gap)
    print(f"\nthe same method against `gap_wp` over {deep_bands} bands: {deep:.0f}")
    print("  (was published as 6096 when the deep gap was the axis)")


if __name__ == "__main__":
    main()
