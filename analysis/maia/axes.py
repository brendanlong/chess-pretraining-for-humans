"""Score Maia-derived difficulty axes against the one the bank uses — and show
why the score can't be taken at face value.

The estimator is `trainer.fit_difficulty --axes`: order items by a candidate
measure of hardness and ask how far the 75th percentile of *erring player
strength* moves from the easiest ninth to the hardest. Scale-free, so measures
in different units are comparable, and bootstrapped here because the differences
are small enough that an argmax on this metric is not by itself evidence.

That target is the trap this script exists to make visible. `spread` scores
against `mover_elo`, and one of the two moves it is handed is the move that
player chose — so a term that inverts Maia into a rating classifier over the
played move scores extremely well while saying nothing about difficulty. The
per-move decomposition below separates the two, and `control.py` prices the
tautology on positions that contain no error at all.

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


def logp(pol, elo, items, field):
    p = pol[elo]
    return np.array([math.log(max(p[r["id"]][field], PROB_FLOOR)) for r in items])


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

    # What `trainer.fit_difficulty.rows` takes, and for the same reasons: a real
    # human's error, and not one mined at a chosen gap band. It also drops
    # unlearnable items and ladder-less ones; `export.py` carries
    # `solution_depth`, which is what learnability is read off.
    base = [
        r
        for r in read(args.items)
        if r["distractor_source"] == "game"
        and r["mined_untargeted"]
        and r["mover_elo"]
        and r["solution_depth"]
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
            f"  {label:<48} {spread(elo, hardness):>+8.1f}  "
            f"{paired.mean():>+7.1f} [{np.quantile(paired, 0.025):>+6.1f},"
            f"{np.quantile(paired, 0.975):>+6.1f}]"
        )

    hi, lo = max(pol), min(pol)
    # How much more probable each move gets as the modelled player strengthens.
    on_best = logp(pol, hi, base, "p_best") - logp(pol, lo, base, "p_best")
    on_played = logp(pol, hi, base, "p_dist") - logp(pol, lo, base, "p_dist")
    w = args.weight

    print(f"n = {len(base)} untargeted, learnable, game-source items; {DRAWS} resamples")
    print(f"  {'axis':<48} {'spread':>8}  paired vs shallow gap")
    row("shallow gap (what the bank uses)", -shallow)
    at_hi = logp(pol, hi, base, "p_best") - logp(pol, hi, base, "p_dist")
    row(f"maia-{hi} log-odds between the two moves", -at_hi)
    print("\n  the elo-gradient, and the two halves it is a difference of:")
    row(f"gradient = on-best - on-played ({hi} - {lo})", -(on_best - on_played))
    row(f"  + shallow gap, w={w}", -(w * z(on_best - on_played) + (1 - w) * z(shallow)))
    row("on-best only  (the discrimination the app trains)", -on_best)
    row(f"  + shallow gap, w={w}", -(w * z(on_best) + (1 - w) * z(shallow)))
    row("on-played only  (never looks at the best move)", on_played)
    row(f"  + shallow gap, w={w}", -(w * z(-on_played) + (1 - w) * z(shallow)))

    print(
        "\n  The gradient's score is the on-played half. That half is Maia read"
        "\n  backwards — a rating classifier over the move this player chose — and"
        "\n  `spread` scores against that player's rating, so it is scoring itself."
        f"\n  r(on-played, mover_elo) = {np.corrcoef(on_played, elo)[0, 1]:+.3f}, "
        f"r(on-best, mover_elo) = {np.corrcoef(on_best, elo)[0, 1]:+.3f}."
        "\n  Run control.py for what the same statistic scores where nobody erred."
    )


if __name__ == "__main__":
    main()
