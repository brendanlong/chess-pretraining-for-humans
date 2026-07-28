# DESIGN — how the app is put together

Three parts: an offline data pipeline, a small API server, a static
frontend. Data flows one way:

    Lichess dump → candidates → labeled items (SQLite) → trials → responses

## Data pipeline (offline, run occasionally)

- **Mining** (`trainer/mine.py`) streams the head of a monthly Lichess PGN
  dump — no full download — keeps server-analyzed blitz-and-slower games,
  and emits decision points where the played move lost a calibrated slice
  of win probability.
- **Labeling** (`trainer/label.py`) runs local Stockfish per candidate:
  a deep multipv search provides ground truth (best move, both evals,
  8-ply lines for both moves); a shallow pass implements the learnability
  filter; the win-probability gap seeds the item's difficulty rating.
  Gap-range flags allow mining specific difficulty bands.
  (`trainer/backfill_pvs.py` retrofits lines onto older items.)

## Server (`trainer/server.py`)

FastAPI over a single SQLite file (WAL; shared safely with a running
labeler). Three endpoints: next trial, answer, stats. Selection picks an
unseen learnable item near the rating where the user's expected score is
80%. Answers move user and item Elo ratings; new users run a calibration
staircase first (start low, big steps, halve on miss). All responses are
recorded with timing for later analysis.

## Frontend (`web/`)

Vanilla JS on vendored chessground; no build step, no client-side chess
logic — the server precomputes SAN and per-ply FENs so the client only
renders. Candidate moves are drawn as arrows; answers by tap or keyboard;
the reveal shows evals in centipawns and win probability, auto-plays the
chosen move's engine line (switchable to the other, steppable, speed
configurable), and can copy the position + both lines as plain text for
pasting into an assistant.

Layout is mobile-first: a slim header (rating, recent accuracy, user,
settings) over a single column, with the space under the board swapping
between the choice buttons and the reveal — verdict and rating delta,
then replay controls with a primary Next, then the two engine lines as
tappable cards; secondary detail sits below the fold. Desktop places the
same panel beside the board. A settings drawer holds replay speed, a
per-name user switcher (URL param overrides localStorage; real accounts
later), and session/debug counters like the fresh-item count.

## Storage

One SQLite database: `items` (positions, moves, evals, lines, difficulty),
`users` (rating, calibration state), `responses` (every answer, timed,
with rating snapshots). The item bank is disposable and rebuildable from
the pipeline; responses are the experimental record.
