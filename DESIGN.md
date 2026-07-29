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
in memory rather than captcha'd. Login is throttled **per account**, not per
address: what an attacker is guessing at is one account's password, and
rotating addresses is a line of script, while several real users share one
address routinely. The price is that a known account can be held locked while
someone keeps guessing at it — inherent to per-account throttling, with the
short window as the mitigation. Signup is throttled per address only because
there is no account to key on yet; per-IP volume really belongs in a reverse
proxy, which sees the true client, is shared across workers and survives a
restart, so treat that counter as insurance rather than a defence. Counters
are charged before the slow work and never refunded — read-before/write-after
is what a burst walks past, and per-outcome refunds need every exit path to be
right. argon2 runs outside the database lock, under a concurrency cap because
it is memory-hard by design (unbounded parallelism is an out-of-memory button)
and with a short wait rather than an unbounded one, because sync endpoints
share a fixed thread pool and a caller merely waiting still holds a thread the
trial flow needs. Because arriving is
enough to mint a guest, guests that answered nothing and went cold are swept
periodically; anything with a response or a password is never touched.

Deletion runs on the same reasoning as signup, in reverse: a signed-in session
is the proof of ownership, which is what makes an in-app button the primary
path rather than an email thread — the optional email is never verified, so for
most accounts there is no address a request could arrive from. The password is
re-entered anyway, because a shared browser holds the session; the attempt
spends the same per-account budget as a login, since it checks a password and
would otherwise be an unmetered guessing oracle. Guests have no password and so
can't be authenticated at all: clearing the cookie is the only deletion
available on a row nobody can point at. The write erases responses and sessions
before the `users` row they reference, all in one transaction — a half-deleted
account is a live session pointing at nothing. `trainer/account.py` is the
operator's way in for what the app can't reach: putting a password on a
pre-account `?user=` profile, and deleting a row that has no password to
re-enter.

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
account controls (sign up / sign in / sign out / delete, reached from the
header chip), session/debug counters like the fresh-item count, and the legal
links. Delete is two deliberate steps behind a password field, and collapses
again whenever the drawer or the account changes — a destructive control
should never be found already armed.

`terms.html` and `privacy.html` are plain pages beside `index.html`,
sharing its stylesheet, so they ship and version with the app instead of
living in a CMS. They are reachable three ways, because the
notice has to reach a guest who never opens either: a page footer that
says what the answers are for, the signup form's agreement line, and the
drawer's About row.

## Storage

One SQLite database: `items` (positions, moves, evals, lines, difficulty),
`users` (rating, calibration state, optional credentials), `sessions`
(hashed cookie tokens), `responses` (every answer, timed, with rating
snapshots). The item bank is disposable and rebuildable from the pipeline;
responses are the experimental record.

## Deployment (`deploy/`, `terraform/`)

One Fly machine with the database on a volume — SQLite has one writer, so a
second machine would be a second fork of the history rather than redundancy.
The image carries the server only; Stockfish and zstd belong to the pipeline,
which stays on a laptop. Litestream supervises uvicorn and streams the file to
S3 continuously, because a volume is one disk on one host and `responses`
can't be regenerated from anything. AWS holds the backup bucket, Litestream's
IAM user, and the DNS record, in Terraform; Fly's own provider is archived, so
that side is `fly.toml` and `flyctl`.

The consequence worth naming is that a refreshed item bank can't arrive as a
file: the bank and the record share a file, so `trainer/push_items.py` carries
items across as their own database and merges them in, matching on position
rather than on row id. Positions already present are skipped — relabelling an
item under the answers already given to it would make those answers
uninterpretable.
