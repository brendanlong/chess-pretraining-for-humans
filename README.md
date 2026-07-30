# Chess Pretraining

A pairwise move-discrimination trainer: you're shown a real position from a
real game and two candidate moves, and you pick the better one. Stockfish
is the ground truth, feedback is immediate, and difficulty adapts to hold
you near 80% accuracy.

- **[SPEC.md](SPEC.md)** — what this is trying to do, and the invariants.
- **[DESIGN.md](DESIGN.md)** — how the app is put together.

## Running it

Requires [uv](https://docs.astral.sh/uv/), a `stockfish` binary on PATH,
and `zstd`.

```bash
# 1. Mine candidate positions from a Lichess monthly dump (streams the
#    head, no need to download 30GB)
curl -s -r 0-150000000 https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst \
  | zstdcat 2>/dev/null \
  | uv run python -m trainer.mine --max-candidates 2500 > data/candidates.jsonl

# 2. Label with Stockfish — builds data/items.db
uv run python -m trainer.label data/candidates.jsonl

# 3. Serve
uv run uvicorn trainer.server:app --host 0.0.0.0 --port 8000
```

The database is `data/items.db` unless `TRAINER_DB` says otherwise — which is
how the container finds it on its volume.

Labeling is the slow step and its defaults are laptop-sized. On a bigger box
raise `--workers` toward the core count before `--threads`: Stockfish scales
better as independent searches than as one wide one.

### Keeping the bank full

Selection can't fail — it takes the nearest items whatever their difficulty —
so a thin patch in the bank shows up as users being served the wrong difficulty,
silently. `trainer.supply` is what makes it visible:

```bash
uv run python -m trainer.supply           # per user rating: what its band holds
uv run python -m trainer.supply --gaps    # the same shortfall, in mining units
```

Refilling a thin band is the ordinary two steps with a gap window on each. The
server evals mining filters on and the deep evals labeling scores with track
each other closely, so a mined window really does aim at a difficulty — it just
doesn't bound it, since the deep search scores some positions wider.

```bash
curl -s -r 0-4000000000 https://database.lichess.org/standard/lichess_db_standard_rated_2026-06.pgn.zst \
  | zstdcat 2>/dev/null \
  | uv run python -m trainer.mine --min-gap-wp 0.20 --max-gap-wp 0.25 \
      --max-candidates 2000 > data/band.jsonl
uv run python -m trainer.label data/band.jsonl --workers 20 --threads 1
```

Mining is cheap next to labeling, so mine a window generously and label what
the shortfall asks for. Different months are independent streams and can run at
once; `trainer.label` skips positions the bank already holds, so overlapping
mines cost nothing but disk.

## Using it

Press <kbd>1</kbd>/<kbd>2</kbd> (or tap) to answer; the chosen move's
engine line then auto-plays on the board. <kbd>1</kbd>/<kbd>2</kbd> or the
tabs switch lines, <kbd>&larr;</kbd>/<kbd>&rarr;</kbd> or the buttons under
the board step through, ⟲ returns to the choice, ⚙ sets replay speed.
<kbd>space</kbd> for the next trial. "Copy for Claude" exports the position
and both lines as text to ask an assistant about.

You start answering immediately as an anonymous guest — the server issues
the identity, no signup. Signing up (header chip → Settings → Account)
attaches a username and password to the guest you've been playing on, so
nothing resets; the email is optional, unverified, and only there for a
future password reset. Signing in on another device picks the account up.

The same drawer deletes the account, confirmed with your password: the row,
its sessions, and all of its responses go in one transaction. That is the
one place the app destroys research data on purpose — see the privacy policy.

`trainer.account` is the operator's way into rows the app can't reach: the
ones that predate accounts (the old `?user=name` profiles) have no guest
session to claim them and no password to re-enter.

```bash
uv run python -m trainer.account list
uv run python -m trainer.account set-password brendan
uv run python -m trainer.account delete brendan   # asks for the name back
```

Exposing it beyond your own machine — a reverse proxy in front, TLS, the
`CLIENT_IP_HEADER` the rate limiters need behind one — is covered in
[deploy/README.md](deploy/README.md).

## Brand assets

The favicons, the app icons, and the 1200×630 social card in `web/` are all
generated and committed. Regenerate them after editing the art:

```bash
uv run --group assets python scripts/generate_assets.py
```

Playwright is not part of the app, so it sits in its own `assets` dependency
group that neither `uv sync` nor the Docker build installs. Setting it up is
two steps — the group, then the browser it drives:

```bash
uv sync --group assets
uv run --group assets playwright install chromium
```

`optipng` also has to be on PATH (`apt install optipng`): the script recompresses
each PNG it writes, losslessly, and `tests/test_assets.py` fails on any PNG in
`web/` that arrived without that — including one added by hand.

The icons come out of a bishop silhouette defined in that script; the card is
a screenshot of `scripts/social-preview.html`, a hand-written wireframe of the
trial screen, so nothing has to be running to rebuild it. The icons are pure
geometry and regenerate byte-identically, but the card renders `system-ui` and
whatever Chromium you have, so it can come out slightly different on another
machine — check the diff is one you meant.

Two pieces of borrowed art: the bishop is public domain (CC0), chosen over the
vendored bishop precisely because a logo shouldn't carry obligations. The
board pieces on the card are the cburnett set by Colin M.L. Burnett, CC BY-SA
— the same art `web/vendor/chessground.cburnett.css` already draws the board
with, credited on the card itself because the card gets shared detached from
the site.

## Deploying it

One Fly machine, SQLite on a volume, Litestream replicating to S3, DNS and the
bucket in Terraform. Bootstrap and runbooks in **[deploy/README.md](deploy/README.md)**;
the AWS side in [terraform/README.md](terraform/README.md).

Mining and labeling stay local, so a refreshed bank is pushed to the
deployment rather than deployed with it — `./deploy/push-items.sh` merges the
new positions in without touching the responses that share the file.

## Checks

CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)) runs all of these
on every push and PR; none of them need Stockfish.

```bash
uv run pytest              # tests
uv run ruff check .        # lint (--fix to autofix)
uv run ruff format .       # format
uv run pyright             # types
npm ci && npm run lint     # eslint over web/ (vendor excluded)
```
