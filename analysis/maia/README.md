# Maia probe

Offline only. Nothing here is imported by the app, nothing here writes to the
bank, and none of it is in the deployment's dependencies — it needs torch and a
GPU, which the server has no business carrying.

It answers one question: **how often is Maia's move Stockfish's move, on our
positions, and does that depend on how Maia is configured?**
[Maia](https://github.com/CSSLab/maia-chess) is a human-imitation policy
conditioned on rating, so it is the closest thing available to asking what a
club player would do with an item. CALIBRATION.md carries the answer and the
one thing that was tried beyond it and rejected.

## Setup

Needs its own venv: maia2 pins torch, the app pins neither.

```bash
uv venv --python 3.12 /tmp/maia-venv
VIRTUAL_ENV=/tmp/maia-venv uv pip install maia2 numpy
```

`--family maia2` downloads its ~280 MB checkpoint on first use, into the
directory it is run from. `--family maia1` additionally wants an `lc0` binary,
the nine `maia-{1100..1900}.pb.gz` weights from the maia-chess repo, and the
`maia1_lc0.py` policy wrapper, all under `~/.local/share/maia` — `policy.py`
documents the layout it expects. lc0 has to be built from source.

## Running it

```bash
python export.py data/items.db items.jsonl
python policy.py items.jsonl maia2.jsonl                  # ~40s for the bank
python policy.py items.jsonl maia1.jsonl --family maia1   # ~3min
python agreement.py items.jsonl rapid=maia2.jsonl maia1=maia1.jsonl
```

`policy.py --speed blitz` and `--opponent-elo` are the other two configuration
knobs; pass each run to `agreement.py` as another `label=path` to compare them.
`control.py`'s docstring carries the control set, which the bank's own number
can't be read without.

## Two traps, both of which cost a wrong number before they were found

**lc0 names castling king-takes-rook**, so `e1g1` is simply absent from its
output. A `dict.get(move, 0.0)` therefore reads the position's most popular
move as probability zero — it hit every one of the 808 items whose best move is
castling, where maia2 gives a median probability of 0.34. `policy.py` re-keys
lc0's whole policy through python-chess's legal move list instead, so a notation
the two disagree about raises rather than reading as nil.

**lc0 prints a prior as two decimals of a percent**, so anything genuinely small
arrives as a flat zero, and a log-odds taken over an unfloored zero is an
outlier of arbitrary size. `policy.PROB_FLOOR` is the resolution both families
are held to. Between them these two moved the apparent correlation between the
families from 0.35 to 0.86.

Neither touches the agreement numbers, which are argmax comparisons — as is
lc0's `--policy-softmax-temp`, whose default of 1.36 rescales every prior it
prints but cannot reorder them (measured: same top move on 300/300 positions at
temperature 2.5).
