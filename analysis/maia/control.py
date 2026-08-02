"""The same agreement measurement on positions nobody was selected into.

The bank's number is unreadable on its own: it exists to hold positions a human
got wrong, so a low agreement rate there could be a fact about Maia or a fact
about our sampling. Positions mined with the gap window opened all the way tell
the two apart.

    curl -s -r 0-220000000 <lichess dump> | zstdcat \
      | uv run python -m trainer.mine --min-gap-wp -1 --max-gap-wp 1 \
          --max-candidates 4000 > control-candidates.jsonl
    python label_control.py control-candidates.jsonl control.jsonl
    python policy.py control.jsonl maia2-control.jsonl --moves best_uci played_uci
    python control.py control.jsonl maia2-control.jsonl

`--min-gap-wp -1` rather than 0 because a played move that *gained* win
probability is as unselected as one that lost a little, and excluding it would
put the error filter back in a milder form.
"""

import argparse

import numpy as np
from agreement import read

# What `trainer.mine` admits by default: below it the played move was fine.
MINING_FLOOR = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("policy")
    args = ap.parse_args()

    ctrl = dict(enumerate(read(args.control)))
    pol = {}
    for r in read(args.policy):
        pol.setdefault(r["elo"], {})[r["id"]] = r
    mid = sorted(pol)[len(pol) // 2]

    played_best = [i for i in ctrl if ctrl[i]["played_uci"] == ctrl[i]["best_uci"]]
    groups = [
        ("all control positions", list(ctrl)),
        ("  where the human played the best move", played_best),
        (
            f"  where the human erred (gap >= {MINING_FLOOR})",
            [i for i in ctrl if ctrl[i]["gap_wp_mined"] >= MINING_FLOOR],
        ),
    ]
    print(f"maia-{mid}'s top move is Stockfish's best move:\n")
    for label, sel in groups:
        a = np.mean([pol[mid][i]["top"] == ctrl[i]["best_uci"] for i in sel])
        print(f"  {label:<42} {a:>7.1%}  n={len(sel)}")
    print(
        f"\n  The humans who were actually at the board played it on "
        f"{len(played_best) / len(ctrl):.1%}\n  of these positions."
    )


if __name__ == "__main__":
    main()
