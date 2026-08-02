"""How often Maia's move is Stockfish's, and what that depends on.

The one question with a clean answer. `policy.py` already records, per position
and per rating, which move Maia ranks first and where it puts the item's two;
this counts them.

Every run is one configuration, so pass several to compare:

    agreement.py items.jsonl rapid=maia2.jsonl blitz=maia2-blitz.jsonl \
                             maia1=maia1.jsonl

With `--best-field best_uci` it reads a control file the same way, which is the
only way the bank's number means anything — the bank exists to hold positions a
human got wrong, so its agreement rate is a fact about our sampling until an
unselected set is standing next to it.
"""

import argparse
import json

import numpy as np


def read(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("policies", nargs="+", metavar="LABEL=PATH")
    ap.add_argument("--best-field", default="best_uci")
    ap.add_argument("--played-field", default="distractor_uci")
    args = ap.parse_args()

    # A control file has no ids; `policy.py` numbers those by position, so the
    # same fallback here keeps the two sides of the join agreeing.
    items = {r.get("id", i): r for i, r in enumerate(read(args.items))}
    runs = {}
    for spec in args.policies:
        label, _, path = spec.partition("=")
        rows = read(path or label)
        by_elo = {}
        for r in rows:
            by_elo.setdefault(r["elo"], {})[r["id"]] = r
        runs[label] = by_elo

    elos = sorted(set.intersection(*[set(v) for v in runs.values()]))
    ids = sorted(set.intersection(*[set(v[elos[0]]) for v in runs.values()]))
    print(f"{len(ids)} positions\n")

    print("Maia's top move is Stockfish's best move:\n")
    print(f"  {'maia rating':>12}" + "".join(f"{k:>12}" for k in runs))
    for e in elos:
        cells = "".join(
            f"{np.mean([runs[k][e][i]['top'] == items[i][args.best_field] for i in ids]):>11.1%} "
            for k in runs
        )
        print(f"  {e:>12}{cells}")

    print("\nWhere Stockfish's best move sits in Maia's ranking:\n")
    header = f"{'top 1':>8} {'top 3':>8} {'top 5':>8} {'top 10':>8} {'median rank':>12}"
    print(f"  {'config':>12} {header}")
    for k, by_elo in runs.items():
        mid = elos[len(elos) // 2]
        ranks = np.array([by_elo[mid][i]["rank_best"] for i in ids])
        cells = "".join(f"{np.mean(ranks < c):>7.1%} " for c in (1, 3, 5, 10))
        print(f"  {k:>12} {cells} {np.median(ranks):>11.0f}")
    print(f"  (at maia-{elos[len(elos) // 2]}; rank 0 is Maia's own first choice)")

    if args.played_field in items[ids[0]]:
        print("\nAnd how often it is the move the human actually played:\n")
        print(f"  {'maia rating':>12}" + "".join(f"{k:>12}" for k in runs))
        for e in elos:
            cells = ""
            for k in runs:
                hit = [runs[k][e][i]["top"] == items[i][args.played_field] for i in ids]
                cells += f"{np.mean(hit):>11.1%} "
            print(f"  {e:>12}{cells}")


if __name__ == "__main__":
    main()
