"""What Maia looks like on positions nobody was selected into.

Two things need a control, and it is the same one. Maia's agreement rate with
Stockfish over the bank is unreadable on its own, because the bank exists to
hold positions a human got wrong. And the elo-gradient's score on `spread` is
unreadable on its own, because the metric's target is the strength of the
player whose move is one of its two inputs.

Both become readable against positions mined with the gap window opened all the
way, so that "a human blundered here" is switched off:

    curl -s -r 0-220000000 <lichess dump> | zstdcat \
      | uv run python -m trainer.mine --min-gap-wp -1 --max-gap-wp 1 \
          --max-candidates 4000 > control-candidates.jsonl
    python label_control.py control-candidates.jsonl control.jsonl
    python policy.py control.jsonl maia2-control.jsonl --moves best_uci played_uci
    python control.py control.jsonl maia2-control.jsonl [items.jsonl maia2.jsonl]

`--min-gap-wp -1` rather than 0 because a played move that *gained* win
probability is as unselected as one that lost a little, and excluding it would
put the error filter back in a milder form.
"""

import argparse
import math
from collections import defaultdict

import numpy as np
from axes import QUANTILE, read, spread
from policy import PROB_FLOOR

# What `trainer.mine` admits by default: below it the played move was fine.
MINING_FLOOR = 0.03


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("control_policy")
    ap.add_argument("items", nargs="?")
    ap.add_argument("items_policy", nargs="?")
    args = ap.parse_args()

    ctrl = dict(enumerate(read(args.control)))
    pol = defaultdict(dict)
    for r in read(args.control_policy):
        pol[r["elo"]][r["id"]] = r
    mid = sorted(pol)[len(pol) // 2]

    print("Maia's top move is Stockfish's best, by how selected the position is\n")
    played_best = [i for i in ctrl if ctrl[i]["played_uci"] == ctrl[i]["best_uci"]]
    groups = [
        ("all control positions", list(ctrl)),
        ("  the human played the engine's best move", played_best),
        (
            f"  the human erred (gap >= {MINING_FLOOR})",
            [i for i in ctrl if ctrl[i]["gap_wp_mined"] >= MINING_FLOOR],
        ),
    ]
    if args.items:
        items = {r["id"]: r for r in read(args.items)}
        ipol = defaultdict(dict)
        for r in read(args.items_policy):
            ipol[r["elo"]][r["id"]] = r
        agree = np.mean([ipol[mid][i]["top"] == items[i]["best_uci"] for i in ipol[mid]])
        print(
            f"  {'the item bank (selected for a human erring)':<44} {agree:>7.1%}"
            f"  n={len(ipol[mid])}"
        )
    for label, sel in groups:
        a = np.mean([pol[mid][i]["top"] == ctrl[i]["best_uci"] for i in sel])
        print(f"  {label:<44} {a:>7.1%}  n={len(sel)}")
    print(
        f"\n  maia-{mid} is the level shown. For scale, the humans who were actually"
        f"\n  at the board played Stockfish's move on "
        f"{len(played_best) / len(ctrl):.1%} of these positions."
    )

    print("\n\nWhat the elo-gradient's winning half scores where nobody erred\n")
    hi, lo = max(pol), min(pol)
    for label, sel in groups:
        sel = [i for i in sel if ctrl[i]["mover_elo"]]
        e = np.array([ctrl[i]["mover_elo"] for i in sel], float)
        on_played = np.array(
            [
                math.log(max(pol[hi][i]["p_dist"], PROB_FLOOR))
                - math.log(max(pol[lo][i]["p_dist"], PROB_FLOOR))
                for i in sel
            ]
        )
        print(
            f"  {label:<44} {spread(e, on_played):>+7.1f}  "
            f"r={np.corrcoef(on_played, e)[0, 1]:+.3f}  n={len(sel)}"
        )
    print(
        f"\n  Same statistic, same estimator, {QUANTILE:g} quantile. The row for positions"
        "\n  where the human played the best move contains no error, no distractor and"
        "\n  nothing to discriminate — so whatever it scores there is the metric"
        "\n  reading the mover's rating off the mover's move, and has to come off"
        "\n  the gradient's score on the bank before any of it is difficulty."
    )


if __name__ == "__main__":
    main()
