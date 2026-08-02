"""The question the other way round: does Maia find our *answers* plausible?

`agreement.py` asks whether Maia picks Stockfish's move. This asks the weaker
and more useful thing — whether Stockfish's move is one a human of a given
strength would consider at all. An item whose answer is invisible to the player
being served can still be a fair forced choice, but it is not one they could
have reasoned their way to, and a bank full of them would be training
recognition of engine moves rather than of chess.

Maia's probability for the best move is the estimate, and its rank among the
legal moves is the estimate that survives a position having forty of them.

    plausibility.py items.jsonl maia2.jsonl [control.jsonl maia2-control.jsonl]

The control is what says whether a number here is about our selection or about
chess; `control.py` documents how it is made.
"""

import argparse
from collections import defaultdict

import numpy as np
from agreement import read

# Below this a move is one the modelled player would essentially never find.
INVISIBLE = 0.01
BANDS = 400


def load(items_path, policy_path):
    items = {r.get("id", i): r for i, r in enumerate(read(items_path))}
    pol = defaultdict(dict)
    for r in read(policy_path):
        pol[r["elo"]][r["id"]] = r
    return items, pol


def summarise(pol, ids, elos):
    out = []
    for e in elos:
        rows = [pol[e][i] for i in ids]
        out.append(
            (
                np.median([r["p_best"] for r in rows]),
                np.mean([r["p_best"] < INVISIBLE for r in rows]),
                # Uniform over the legal moves is the only floor that means the
                # same thing in a position with 8 moves and one with 50.
                np.mean([r["p_best"] < 1 / r["n_legal"] for r in rows]),
                np.mean([r["rank_best"] >= 5 for r in rows]),
            )
        )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("policy")
    ap.add_argument("control", nargs="?")
    ap.add_argument("control_policy", nargs="?")
    args = ap.parse_args()

    items, pol = load(args.items, args.policy)
    elos = [min(pol), max(pol)]
    labels = ["median P(best)", f"P(best) < {INVISIBLE}", "below uniform", "outside top 5"]

    sets = [("item bank", list(pol[elos[0]]), pol)]
    if args.control:
        _, cpol = load(args.control, args.control_policy)
        sets.append(("control", list(cpol[elos[0]]), cpol))

    print("Is Stockfish's best move one a human would consider?\n")
    head = "".join(f"{s + ' ' + str(e):>16}" for s, _, _ in sets for e in elos)
    print(f"  {'':<24}{head}")
    cols = [summarise(p, ids, elos) for _, ids, p in sets]
    for j, lab in enumerate(labels):
        cells = "".join(
            f"{c[k][j]:>16.3f}" if j == 0 else f"{c[k][j]:>15.1%} "
            for c in cols
            for k in range(len(elos))
        )
        print(f"  {lab:<24}{cells}")

    if "rating" not in items[next(iter(items))]:
        return
    lo = elos[0]
    print(f"\nAgainst the bank's own difficulty scale, at maia-{lo}.")
    print("The scale is built from engine measurements and has never seen Maia,")
    print("so this is one oracle checking the other.\n")
    print(
        f"  {'item rating band':<20}{'n':>7}{'median P(best)':>16}{f'P<{INVISIBLE}':>10}"
        f"{'outside top 5':>15}"
    )
    ratings = [items[i]["rating"] for i in pol[lo]]
    for band in range(0, int(max(ratings)) + BANDS, BANDS):
        sel = [i for i in pol[lo] if band <= items[i]["rating"] < band + BANDS]
        if len(sel) < 50:
            continue
        rows = [pol[lo][i] for i in sel]
        print(
            f"  {band:>5}-{band + BANDS:<14}{len(sel):>7}"
            f"{np.median([r['p_best'] for r in rows]):>16.3f}"
            f"{np.mean([r['p_best'] < INVISIBLE for r in rows]):>10.1%}"
            f"{np.mean([r['rank_best'] >= 5 for r in rows]):>15.1%}"
        )
    print("\n  Selection serves a beginner from the top rows.")


if __name__ == "__main__":
    main()
