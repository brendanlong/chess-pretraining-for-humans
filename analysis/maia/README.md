# Maia probe

Offline only. Nothing here is imported by the app, nothing here writes to the
bank, and none of it is in the deployment's dependencies — it needs torch and a
GPU, which the server has no business carrying.

What it is for: every measurement the bank holds comes from a search engine,
which can say what is *true* but not what is *hard*. [Maia](https://github.com/CSSLab/maia-chess)
is a human-imitation policy conditioned on rating, so it can say what a player
of a stated strength would probably play — the denominator CALIBRATION.md says
the error-only sample doesn't have.

Read that file's section on this before drawing anything from `axes.py`. The
short version is that a Maia statistic over the *played* move inverts to a
rating classifier, `spread` scores against the rating of the player who played
it, and the two multiply into a very large number that is not difficulty.
`axes.py` splits every measure per move for that reason, and `control.py`
exists to price the part that survives on positions containing no error.

## Setup

Needs its own venv: maia2 pins torch, the app pins neither.

```bash
uv venv --python 3.12 /tmp/maia-venv
VIRTUAL_ENV=/tmp/maia-venv uv pip install maia2 numpy
```

`--family maia2` downloads its ~280 MB checkpoint on first use. `--family maia1`
additionally wants an `lc0` binary, the nine `maia-{1100..1900}.pb.gz` weights
from the maia-chess repo, and the `maia1_lc0.py` policy wrapper, all under
`~/.local/share/maia` — `policy.py` documents the layout it expects. lc0 has to
be built from source, and its `--policy-softmax-temp` defaults to 1.36, which
silently rescales every prior it prints.

## Running it

```bash
python export.py data/items.db items.jsonl
python policy.py items.jsonl maia2.jsonl --family maia2   # ~40s for the bank
python policy.py items.jsonl maia1.jsonl --family maia1   # ~3min
python axes.py items.jsonl maia2.jsonl
```

`control.py`'s docstring carries the rest — the control set is mined and labeled
separately, and everything it prints needs it.

## Reading the output

`axes.py` scores candidate measures the way `trainer.fit_difficulty --axes` does,
so its numbers are directly comparable with the ones in CALIBRATION.md. The
differences it reports are small relative to the resampling noise, which is why
it bootstraps rather than printing an argmax; treat a column of point estimates
from it as unreadable without the paired interval beside it.

Two traps, both of which cost a wrong published number before they were found.

**lc0 names castling king-takes-rook**, so `e1g1` is simply absent from its
output. A `dict.get(move, 0.0)` therefore reads the position's most popular
move as probability zero — it hit 4.7% of the bank, every item whose best move
is castling, and maia2 gives those same moves a median probability of 0.34.
`policy.py` re-keys lc0's whole policy through python-chess's legal move list
instead, so a notation the two disagree about raises rather than reading as nil.

**lc0 prints a prior as two decimals of a percent**, so anything genuinely small
arrives as a flat zero and an unfloored log-odds over it is an outlier of
arbitrary size. `policy.PROB_FLOOR` is the resolution both families are held to.
Between them these two moved the apparent correlation between the families from
0.35 to 0.86 — the floor is worth about half of that and the castling fix the
other half, which is why attributing it all to resolution was itself a mistake.
