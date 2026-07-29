# chess-pretraining-for-humans

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

Behind a reverse proxy, terminate TLS there (the session cookie is marked
`Secure` whenever the request arrives over https) and run uvicorn with
`--proxy-headers` and a trusted `--forwarded-allow-ips`, so the signup limit
sees real client addresses rather than counting the whole site as one. Login
throttling is per account and unaffected either way.

If you expose this publicly, put per-IP request limiting in the proxy
(`limit_req` in nginx, or equivalent) rather than relying on the in-app
counter: the proxy's is shared across workers, survives a restart, and sheds
load before it reaches Python.

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
