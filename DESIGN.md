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

An answer is only accepted for a trial the server actually offered, which
`trainer/trials.py` carries in an HMAC-signed token rather than a row: the answer
payload *is* the answer key, and item ids are small sequential integers, so
otherwise the whole bank reads out by counting and the trial on your own screen
is one request from being looked up. The token names its holder as well as its
item, so a throwaway client can't fetch one for the signed-in client to spend.

Signed, not stored, for two reasons. A row to write it in would have to exist
before the first trial, which is exactly the pressure that used to put a limit in
front of arriving; and state means one pending trial per user, so two tabs fight.
The remaining hole is small and known: a trial served before its owner has any
identity is bound to nobody, so another anonymous caller can redeem it —
recording the answer on *their* row, never accumulating onto a session that keeps
its cookie. That, and the fact that `next`→`answer` reaches every item anyway,
is why enforcement stops here rather than growing a nonce store. What *does*
protect the shared counters is that only first exposures move them, and a
per-address limit on answering.

The signing key comes from `TRIAL_TOKEN_SECRET` (a Fly secret). Rotating it costs
nothing but the trials in flight; unset, the server logs that it made an
ephemeral one, which on a restarting machine means one refused answer per open
tab.

## Identity (`trainer/auth.py`)

Anonymous-first, and lazy: arriving writes nothing. The first *answer* mints a
guest `users` row and an opaque session token in an HttpOnly cookie; no name is
typed and no name is guessable, which the old `?user=` scheme couldn't say. An
earlier version issued that row on arrival, and the cost of it kept surfacing —
crawlers and health checks grew the table, and metering the write was a limit in
front of the first trial. Tying the row to the first answer means the only
unauthenticated write is the one that produces something worth keeping. Signing
up attaches a username, an argon2 password hash, and an optional unverified email
(kept only for a future reset) to *that same row* — or creates it outright, for
someone who signs up before answering anything — so an account is a claim on
history rather than a gate in front of it. Sessions are a table of hashed
tokens, so a database read grants no logins; the token is rotated on every
privilege change and expires both on idleness and absolutely, so a token that
keeps being used still stops being a credential eventually. `SameSite=Lax`
plus `no-store` on every API response is the CSRF-and-shared-cache story, and a
CSP with no allowlist (everything the page loads is ours) is what stops a
future hostile string in mined data from being script.

Rate limiting, not captchas: the cost of a wrong guess here should be a wait,
not a lost signup. Every password check is metered twice, and each half stops
something the other can't. **Per name typed** is what protects one account's
password — rotating addresses is a line of script, while several real users
share one address routinely. **Per address** is what protects the box: argon2
is memory-hard on purpose and only a few run at once, so an unmetered password
check is a way to answer every real user 503, whether or not the name it names
exists. Which is also why the name key is the name *submitted* rather than the
row it resolves to: key on a row id and a name nobody registered has no
counter, so the presence of a 429 becomes the answer to "does this account
exist?" — an enumeration oracle that costs eleven requests and undoes the
careful dummy-hash verify that keeps the *timing* from saying the same thing.
Handing the key space to the caller is the price, which is why the limiter
evicts its least-throttled keys rather than its oldest when full, and why the
per-address budget in front makes filling it expensive.

Per-name throttling means a known account can be held locked by someone who
keeps guessing at it — inherent, with the short window as the mitigation — so
deletion is keyed separately: the one irreversible thing a user might need to
do *because* they're under attack must not be blockable from outside.

Signup is keyed on address alone because there is no account to key on yet, and
so is answering, because that is the only unauthenticated write left and the only
thing that moves `items.attempts`/`correct` — global counters every user's
difficulty targeting reads. The trial binding is no help there: `next`→`answer`
skews an item as well as a bare `answer` did, one request later. That limit sits
on the core loop, where several real users share one address routinely, so it is
set far above any human pace; there is deliberately no limit at all on arriving,
because arriving writes nothing to ration. "Address" behind a proxy means a
header the proxy
overwrites (`CLIENT_IP_HEADER`), never the socket: trusting forwarded headers
makes uvicorn believe the *leftmost* `X-Forwarded-For` entry, which a proxy
appends to rather than replaces, so it is the caller's to invent and a flood
keyed on it would get a fresh counter every request. Per-IP volume really
belongs in a reverse proxy, which sees the true client, is shared across
workers and survives a restart, so treat these counters as insurance rather
than a defence. They are charged before the slow work and never refunded —
read-before/write-after is what a burst walks past, and per-outcome refunds
need every exit path to be right. argon2 runs outside the database lock, under
a concurrency cap because it is memory-hard by design (unbounded parallelism is
an out-of-memory button) and with a short wait rather than an unbounded one,
because sync endpoints share a fixed thread pool and a caller merely waiting
still holds a thread the trial flow needs. Rows that answered nothing and went
cold are swept periodically; anything with a response or a password is never
touched. Since answering is what writes a row, the sweep now mostly tidies
history — the guests the old arrival-minting left behind, and the gap between
minting an identity and recording the answer that earned it.

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
re-enter. Setting a password there signs that account's existing sessions out.
With no reset email yet, that command is the only recovery path there is, so it
has to assume the reason it's being run is that someone else knows the old
password — and rotating the hash while leaving their session live recovers
nothing. It also refuses to act on a name that matches two rows, which a
database missing the case-insensitive unique index allows: guessing there is how
one user's password ends up on another user's history.

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

The palette lives entirely in `:root`, including the dark, saturated
arrow variants the light board needs alongside the light ones the dark
panel needs; `app.js` reads the arrow colours back out of CSS so a brush
can't drift away from the button it is supposed to match.

The candidate pair is blue against orange, which sits on the blue-yellow
axis that red-green deficiencies leave intact, and it holds up even with a
warm night filter stacked on top — that pair has to, because colour is the
only thing tying an arrow to its button. The reveal pair is green against
rose, red pushed far enough toward magenta to keep a blue component;
that is a real improvement over green/red but it is not the same
guarantee, since deuteranopia plus a strong warm filter strips exactly the
component it depends on. No green/red pair can do better on a cream board
— the lightness range both colours have to sit in to stay visible is too
narrow — and the reveal says the same thing in text anyway, so colour
there reinforces rather than carries. `tests/test_palette.py` holds both
pairs to the floors they are meant to clear.

Chessground's global 0.6 dimming of drawn shapes is overridden, because
compositing that far toward the board throws away the separation.

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

The icons and the social-card image are committed files in `web/`, generated
by `scripts/generate_assets.py` screenshotting Chromium — so the art has a
source that can be edited and re-run, and serving it stays a static file
read. The card mock declares no colours of its own: the generator injects
the app's `:root`, because a card advertising the app shouldn't be showing
a palette the app has since moved off. Every page carries its own copy of
the card metadata, since there is no template to share one; a test globs
`web/*.html` and holds each copy against the page it sits on, so a new page
can't ship without it.

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
which stays on a laptop. The entrypoint drops to an unprivileged uid before
starting anything, which is why it isn't a `USER` line: only root can chown a
volume the platform mounts after the build, and Litestream is the supervisor, so
the drop has to wrap it rather than the server. Litestream supervises uvicorn and streams the file to
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
