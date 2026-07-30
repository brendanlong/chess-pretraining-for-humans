# Getting oriented

Before adding or changing features, read:

- **[SPEC.md](SPEC.md)** — goals and invariants (what we want)
- **[DESIGN.md](DESIGN.md)** — architecture (how it's built)
- **[deploy/README.md](deploy/README.md)** — running the live instance
  (bootstrap, runbooks, operational cautions; the AWS half is
  [terraform/README.md](terraform/README.md))

When a change alters behavior, update the relevant file: new goals or
invariants go in SPEC.md, structural changes in DESIGN.md, and anything
an operator does or must not do goes in deploy/README.md. All of them
stay brief, code-free, and non-repeating — if it's obvious from the
code, or already said in one of the others, don't document it. Detailed
"why is it built this way" reasoning lives in comments beside the code
it justifies; DESIGN.md points at it rather than restating it.

Comments document only what a reader needs to know about the *current*
state of the app. State the reason a thing is the way it is, not the
history of how it got that way — no "used to", "an earlier version",
or "no longer" unless the old state left data behind that the code
still handles.

## Working on the code

- Python 3.12 via `uv`; tests with `uv run pytest`. Frontend is
  build-free vanilla JS in `web/` (chessground is vendored).
- Local dev needs a `stockfish` binary on PATH; mining needs `zstd`.
  See README for the pipeline and server commands.
- Any PNG landing in `web/` gets run through `optipng` first — the asset
  generator does it, and a test fails if a committed one skipped it.
- `data/` is gitignored. The item bank (`items.db` items table) is
  rebuildable from the pipeline; the `responses` table is the
  experimental record — never wipe it casually.

## The invariant that bites

Nothing — UI, API payload, timing, rating movement — may reveal which
move is better before the user commits an answer. When touching the
trial flow, check what the client can observe pre-answer. See SPEC.md
for the full list of invariants.
