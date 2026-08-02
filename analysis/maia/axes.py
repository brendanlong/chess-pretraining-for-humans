"""Score Maia-derived difficulty axes against the one the bank uses.

Same estimator as `trainer.fit_difficulty --axes`: order items by a candidate
measure of hardness and ask how far the 75th percentile of *erring player
strength* moves from the easiest ninth to the hardest. Scale-free, so measures
in different units are comparable, and bootstrapped here because the differences
are small enough that an argmax on this metric is not by itself evidence.

    export.py data/items.db items.jsonl
    policy.py items.jsonl maia2.jsonl --family maia2
    axes.py items.jsonl maia2.jsonl
"""

import argparse
import json
import math
from collections import defaultdict

import numpy as np
from policy import PROB_FLOOR

QUANTILE = 0.75
DRAWS = 400


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def spread(elo, hardness, bins=9):
    tails = [np.quantile(c, QUANTILE) for c in np.array_split(elo[np.argsort(hardness)], bins)]
    return float(tails[-1] - tails[0])


def z(x):
    return (x - x.mean()) / x.std()


def log_odds(pol, elo, items):
    p = pol[elo]
    return np.array(
        [
            math.log(max(p[r["id"]]["p_best"], PROB_FLOOR) / max(p[r["id"]]["p_dist"], PROB_FLOOR))
            for r in items
        ]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("policy")
    ap.add_argument(
        "--weight",
        type=float,
        default=0.5,
        help="z-weight on the Maia term when combining with the shallow gap",
    )
    args = ap.parse_args()

    pol = defaultdict(dict)
    for r in read(args.policy):
        pol[r["elo"]][r["id"]] = r

    # The same restriction `trainer.fit_difficulty` takes, and for the same
    # reasons: a real human's error, and not one mined at a chosen gap band.
    base = [
        r
        for r in read(args.items)
        if r["distractor_source"] == "game"
        and r["mined_untargeted"]
        and r["mover_elo"]
        and r["id"] in pol[max(pol)]
    ]
    elo = np.array([r["mover_elo"] for r in base], float)
    shallow = np.array([r["shallow_gap"] for r in base])

    rng = np.random.default_rng(0)
    idx = [rng.integers(0, elo.size, elo.size) for _ in range(DRAWS)]
    baseline = np.array([spread(elo[i], -shallow[i]) for i in idx])

    def row(label, hardness):
        draws = np.array([spread(elo[i], hardness[i]) for i in idx])
        paired = draws - baseline
        print(
            f"  {label:<46} {spread(elo, hardness):>+8.1f}  "
            f"{np.mean(draws > baseline):>5.0%}  "
            f"{paired.mean():>+7.1f} [{np.quantile(paired, 0.025):>+6.1f},"
            f"{np.quantile(paired, 0.975):>+6.1f}]"
        )

    lo_hi, lo_lo = log_odds(pol, max(pol), base), log_odds(pol, min(pol), base)
    # How much more clearly the stronger net prefers the right move than the
    # weaker one does: the item's sensitivity to skill, which is the thing a
    # difficulty axis is trying to name and which no engine measure states.
    gradient = lo_hi - lo_lo
    w = args.weight

    print(f"n = {len(base)} untargeted game-source items, {DRAWS} bootstrap resamples")
    print(f"  {'axis':<46} {'spread':>8}  {'wins':>5}  vs shallow gap alone")
    row("shallow gap (what the bank uses)", -shallow)
    row(f"maia-{max(pol)} log-odds alone", -lo_hi)
    row(f"  + shallow gap, w={w}", -(w * z(lo_hi) + (1 - w) * z(shallow)))
    row(f"elo-gradient ({max(pol)} - {min(pol)}) alone", -gradient)
    row(f"  + shallow gap, w={w}", -(w * z(gradient) + (1 - w) * z(shallow)))
    print(f"\n  gradient vs shallow gap: r = {np.corrcoef(gradient, shallow)[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
