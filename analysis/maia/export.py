"""The item columns the Maia probe reads, as jsonl.

Kept separate from `policy.py` so that the half needing torch and the half
needing the bank don't have to be installed together — the probe runs in its
own venv and never imports the app.
"""

import argparse
import json
import sqlite3

COLUMNS = [
    "id",
    "fen",
    "best_uci",
    "distractor_uci",
    "distractor_source",
    "gap_wp",
    "shallow_gap",
    "solution_depth",
    "rating",
    "mover_elo",
    "ply",
    "mined_untargeted",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("dst")
    args = ap.parse_args()
    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    with open(args.dst, "w") as out:
        for row in conn.execute(f"SELECT {', '.join(COLUMNS)} FROM items ORDER BY id"):
            out.write(json.dumps(dict(zip(COLUMNS, row, strict=True))) + "\n")


if __name__ == "__main__":
    main()
