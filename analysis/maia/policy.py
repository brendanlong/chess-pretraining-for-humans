"""Maia's move probabilities for both candidates of every item, at nine strengths.

Maia is a human-imitation policy: given a position and a rating, it predicts
what a player of that rating would play. Run over the bank it answers the two
questions an engine can't — how likely a human of a stated strength is to pick
each of our two moves, and how that changes with strength.

Both model families, because they disagree enough to be worth having separately:

  --family maia2  one net conditioned on rating (CSSLab 2024). GPU, ~10k evals/s,
                  so the whole bank at nine levels is well under a minute.
  --family maia1  the nine separately-trained KDD'20 nets, driven through lc0 at
                  one node. ~3 minutes for the same sweep, and a genuinely
                  independent replication rather than one model asked twice.

Neither ships with the app. See README.md here for the venv and the weights.

Output is one JSON line per (item, rating). `p_best` and `p_dist` are the
probabilities of the item's two moves; `rank_*` counts legal moves the model
likes strictly better.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ELOS = [1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
MAIA_HOME = Path.home() / ".local/share/maia"

# lc0 prints a move's prior as two decimals of a percent, so anything under
# 5e-5 reads as a flat zero. Both families are floored there — not because
# maia2 needs it, but because a log-odds taken over an unfloored zero is an
# outlier of arbitrary size, and comparing the two families means the coarser
# instrument sets the resolution for both.
PROB_FLOOR = 5e-5


def run_maia2(rows, fa, fb, batch=2048):
    import torch
    from maia2 import model
    from maia2.inference import _masked_softmax, preprocessing
    from maia2.utils import create_elo_dict, get_all_possible_moves, mirror_move

    all_moves = get_all_possible_moves()
    move_idx = {mv: i for i, mv in enumerate(all_moves)}
    buckets = create_elo_dict()

    m = model.from_pretrained(type="rapid", device="gpu").eval()
    dev = next(m.parameters()).device

    boards, masks, black = [], [], []
    for r in rows:
        b, _, _, lm = preprocessing(r["fen"], 1500, 1500, buckets, move_idx)
        boards.append(b)
        masks.append(lm)
        black.append(r["fen"].split(" ")[1] == "b")
    boards, masks = torch.stack(boards), torch.stack(masks)

    # Maia-2's move vocabulary is written from the side-to-move's point of view,
    # so a black-to-move position addresses it through the mirrored move.
    def col(field):
        return torch.tensor(
            [
                move_idx[mirror_move(r[field]) if bl else r[field]]
                for r, bl in zip(rows, black, strict=True)
            ]
        )

    ia, ib = col(fa), col(fb)

    for elo in ELOS:
        code = buckets[f"{elo - elo % 100}-{elo - elo % 100 + 99}"]
        for i in range(0, len(boards), batch):
            bb, mm = boards[i : i + batch].to(dev), masks[i : i + batch].to(dev)
            n = bb.shape[0]
            ee = torch.full((n,), code, dtype=torch.long, device=dev)
            with torch.no_grad():
                logits, _, _ = m(bb, ee, ee)
            p = _masked_softmax(logits, mm).cpu()
            pa = p.gather(1, ia[i : i + n, None]).squeeze(1)
            pb = p.gather(1, ib[i : i + n, None]).squeeze(1)
            top_p, top_i = p.max(dim=1)
            rank_a = (p > pa[:, None]).sum(1)
            rank_b = (p > pb[:, None]).sum(1)
            n_legal = masks[i : i + n].sum(1)
            for j in range(n):
                mv = all_moves[int(top_i[j])]
                yield {
                    "id": rows[i + j]["id"],
                    "elo": elo,
                    "p_best": round(float(pa[j]), 6),
                    "p_dist": round(float(pb[j]), 6),
                    "p_top": round(float(top_p[j]), 6),
                    "top": mirror_move(mv) if black[i + j] else mv,
                    "rank_best": int(rank_a[j]),
                    "rank_dist": int(rank_b[j]),
                    "n_legal": int(n_legal[j]),
                }
        print(f"  maia2 {elo} done", file=sys.stderr, flush=True)


def run_maia1(rows, fa, fb):
    """One lc0 process per rating, in parallel: at a single node lc0 is bound by
    the UCI round trip rather than by the GPU, so nine processes is nine times
    the rate and the CPU backend beats the CUDA one."""
    sys.path.insert(0, str(MAIA_HOME / "scripts"))
    from maia1_lc0 import MaiaPolicy

    def one(elo):
        out = []
        weights = MAIA_HOME / f"weights/maia-{elo}.pb.gz"
        with MaiaPolicy(str(MAIA_HOME / "lc0"), str(weights), backend="eigen") as eng:
            for r in rows:
                p = eng.policy(r["fen"])
                if not p:
                    continue
                top = max(p, key=p.get)
                pa, pb = p.get(r[fa], 0.0), p.get(r[fb], 0.0)
                out.append(
                    {
                        "id": r["id"],
                        "elo": elo,
                        "p_best": pa,
                        "p_dist": pb,
                        "p_top": p[top],
                        "top": top,
                        "rank_best": sum(v > pa for v in p.values()),
                        "rank_dist": sum(v > pb for v in p.values()),
                        "n_legal": len(p),
                    }
                )
        print(f"  maia1 {elo} done", file=sys.stderr, flush=True)
        return out

    with ThreadPoolExecutor(len(ELOS)) as pool:
        for chunk in pool.map(one, ELOS):
            yield from chunk


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("src", help="jsonl with `fen` and the two move fields")
    ap.add_argument("dst")
    ap.add_argument("--family", choices=("maia1", "maia2"), default="maia2")
    ap.add_argument(
        "--moves", nargs=2, default=("best_uci", "distractor_uci"), metavar=("BEST", "DISTRACTOR")
    )
    args = ap.parse_args()

    with open(args.src) as f:
        rows = [json.loads(line) for line in f]
    # The bank keys on `id`; a mined candidate set has none, so index by position.
    for i, r in enumerate(rows):
        r.setdefault("id", i)

    run = run_maia2 if args.family == "maia2" else run_maia1
    with open(args.dst, "w") as out:
        for row in run(rows, *args.moves):
            out.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
