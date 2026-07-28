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
labeler). Trial endpoints: next trial, answer, stats. Selection picks an
unseen learnable item near the rating where the user's expected score is
80%. Answers move user and item Elo ratings; new users run a calibration
staircase first (start low, big steps, halve on miss). All responses are
recorded with timing for later analysis.

## Identity (`trainer/auth.py`)

Anonymous-first. The first request mints a guest `users` row and an opaque
session token in an HttpOnly cookie; no name is typed and no name is
guessable, which the old `?user=` scheme couldn't say. Signing up attaches a
username, an argon2 password hash, and an optional unverified email (kept
only for a future reset) to *that same row*, so an account is a claim on
history rather than a gate in front of it. Sessions are a table of hashed
tokens, so a database read grants no logins; the token is rotated on every
privilege change, and `SameSite=Lax` plus `no-store` on every API response
is the CSRF-and-shared-cache story. Signup and login are rate-limited per IP
in memory rather than captcha'd: a loose per-attempt limit, because an
attempt is what costs us (an argon2 hash, and an answer to "is this name
taken?"), plus a tight limit on accounts actually created. argon2 runs
outside the database lock but under a concurrency cap — it is memory-hard by
design, so unbounded parallelism is an out-of-memory button. Because arriving is
enough to mint a guest, guests that answered nothing and went cold are swept
periodically; anything with a response or a password is never touched.
`trainer/account.py` is the operator's way to put a password on a row the
app can't reach (the pre-account `?user=` profiles).

Two ordering constraints are easy to get wrong. Identity resolution is a
dependency, but it yields a user *id*: FastAPI finishes dependencies before
the endpoint body, so a row read there would be a pre-lock snapshot and
concurrent answers would overwrite each other's ratings. And the session
cookie is applied by middleware (and by the catch-all error handler, which
sits outside it) rather than by the endpoint, so a request that mints a
guest and then fails still hands the identity out instead of orphaning it.
Signup resolves identity in its body rather than through the dependency, for
the same reason in reverse: a throttled request should mint nothing at all.

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
same panel beside the board. A settings drawer holds replay speed, the
account controls (sign up / sign in / sign out, reached from the header
chip), and session/debug counters like the fresh-item count.

## Storage

One SQLite database: `items` (positions, moves, evals, lines, difficulty),
`users` (rating, calibration state, optional credentials), `sessions`
(hashed cookie tokens), `responses` (every answer, timed, with rating
snapshots). The item bank is disposable and rebuildable from the pipeline;
responses are the experimental record.
