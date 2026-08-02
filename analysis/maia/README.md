# Maia probe

Offline only. Nothing here is imported by the app, nothing here writes to the
bank, and none of it is in the deployment's dependencies — it needs torch and a
GPU, which the server has no business carrying.

What it is for: every measurement the bank holds comes from a search engine,
which can say what is *true* but not what is *hard*. [Maia](https://github.com/CSSLab/maia-chess)
is a human-imitation policy conditioned on rating, so it can say what a player
of a stated strength would probably play — the denominator CALIBRATION.md says
the error-only sample doesn't have.

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

`policy.py` also takes a mined candidate file — pass `--moves best_uci played_uci`
— which is how the control set in CALIBRATION.md was measured. A control needs
mining with the gap window opened all the way (`--min-gap-wp -1 --max-gap-wp 1`),
so that the "a human blundered here" filter is off and Maia's agreement rate is
readable as a fact about Maia rather than about our sampling.

## Reading the output

`axes.py` scores candidate measures the way `trainer.fit_difficulty --axes` does,
so its numbers are directly comparable with the ones in CALIBRATION.md. The
differences it reports are small relative to the resampling noise, which is why
it bootstraps rather than printing an argmax; treat a column of point estimates
from it as unreadable without the paired interval beside it.

One trap, since both families are scored together: lc0 prints a move's prior as
two decimals of a percent, so a genuinely small probability arrives as a flat
zero and an unfloored log-odds over it becomes an outlier of arbitrary size.
That alone moved the apparent correlation between the two families from 0.30 to
0.85. `policy.PROB_FLOOR` is the resolution both are held to.
