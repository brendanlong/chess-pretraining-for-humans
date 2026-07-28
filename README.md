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
and both lines as text to ask an assistant about. `?user=name` keeps
separate profiles.

Tests: `uv run pytest`.
