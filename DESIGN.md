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
  8-ply lines for both moves); then one more search, restricted to those two
  moves and started from a cleared hash, ranks the pair at every depth on its
  way back to the same depth. That whole curve is stored (`gap_ladder`), and
  the mean of its shallow end is the item's difficulty (`shallow_gap`),
  through the curve on `rating.difficulty_rating` — whose slope is measured
  from the strength of the humans whose errors the items are, on the half of
  the bank that was mined without aiming at particular gaps, and whose
  reasoning sits beside the constants — and what was tried instead, in
  CALIBRATION.md. Nothing is dropped for being hard;
  only for the two searches disagreeing at full depth, which has nothing to
  teach. Gap-range flags steer mining, but only at the *deep* gap, so an
  order lands across a spread of difficulties rather than in a band.
  (`trainer/backfill_pvs.py` retrofits lines onto older items.) Every row a
  labeler writes carries its ladder and the gap read off it, so nothing
  downstream has to cope with a bank that is only partly measured — `db.connect`
  refuses one outright rather than serving it at an older curve's difficulty.
- **Supply** (`trainer/supply.py`) reports what the bank can serve at each
  user rating, and — because mining aims at the deep gap while difficulty is
  made of the shallow one — where each deep-gap bin's items actually landed,
  rather than an order in items it can no longer honestly state. Why a thin
  band is otherwise invisible is in the module.

## Server (`trainer/server.py`)

FastAPI over a single SQLite file (WAL; shared safely with a running
labeler), one connection per threadpool thread — bound to the thread rather
than checked out and returned, so no two requests can ever hold the same one,
and a connection lives and dies with the worker that opened it. A transaction
belongs to a connection, so sharing one would serialize every request whether
or not it wrote; separate connections let WAL do what it is for, and leave
SQLite to serialize the writers. Endpoints that write take one explicitly and up front
(`writing()`), because a rating is read, computed in Python, and written
back — two overlapping answers that both read first would lose one. How much
of that is scaling headroom is a question of how many cores the machine has,
which is the point: it is now a setting rather than a rewrite.

`writing()` hands out a handle with no commit on it — the same shape as an
ORM's interactive transaction, and for the same reason: the block's outcome is
then the only thing that can end it. Storage helpers take that handle
(`db.Queryable`), so one that wanted to commit could not name the method.

Both objects expose `execute` and nothing else, which is most of what makes
this hold. Neither has a commit to call — outside a transaction a statement
commits itself, and inside one the block owns the ending — so the stray
`commit()` that was harmless in one caller and silently un-atomicked another
can't be written at all, and neither can the `cursor()` that would have got
back to the raw connection. The one thing left to catch at runtime is the case
that looks like it works: running a statement on the ambient connection while a
transaction is open, which is how a block quietly stops being one. That, and
nesting, raise — SQLite has no nested transaction to make either safe.

Trial endpoints: next trial, answer, stats. Selection picks an
unseen learnable item near the rating where the user's expected score is
80%. An answer moves the user's Elo rating and nothing else — item
difficulty is fixed at labeling time, so the `items` row a trial came from
is never written to and no user's answers change what another is served.
New users run a calibration staircase first (start low, big steps, halve on
miss). All responses are recorded with timing and the rating snapshots that
make each trial reconstructible on its own.

An answer is only accepted for a trial the server actually offered, which
`trainer/trials.py` carries in an HMAC-signed token rather than a row: the
answer payload *is* the answer key, and item ids are small sequential
integers, so otherwise the whole bank reads out by counting. Signed rather
than stored so that no row has to exist before the first trial and two tabs
don't fight over one pending-trial slot. The token names its holder and its
item, and says whether the trial was served as a repeat; a token issued
before its holder has any identity is additionally remembered once spent
(`server.anonymous_trial_use`), because redeeming it is what creates the row
that would otherwise notice the replay. The threat model — who a replayed
token serves, what a pre-commit peek is worth, and why enforcement stops
where it does — is spelled out in `trainer/trials.py`'s docstring.

The signing key comes from `TRIAL_TOKEN_SECRET` (a Fly secret). Rotating it
costs nothing but the trials in flight; unset, the server logs that it made
an ephemeral one, which on a restarting machine means one refused answer per
open tab.

## Identity (`trainer/auth.py`)

Anonymous-first, and lazy: arriving writes nothing. The first *answer* mints
a guest `users` row and an opaque session token in an HttpOnly cookie — no
name is typed and no name is guessable — so the only unauthenticated write
is the one that produces something worth keeping, and nothing has to be
rationed in front of the first trial. Signing up attaches a username, an
argon2 password hash, and an optional unverified email (kept only for a
future reset) to *that same row* — or creates it outright, for someone who
signs up before answering anything — so an account is a claim on history
rather than a gate in front of it. Sessions are a table of hashed tokens, so
a database read grants no logins; the token is rotated on every privilege
change and expires both on idleness and absolutely. `SameSite=Lax` plus
`no-store` on every API response is the CSRF-and-shared-cache story, and a
CSP is what stops a hostile string in mined data from being script. Its
allowlist is two named origins, both the hosted page counter — everything
else the page loads is ours.

Abuse control is rate limiting, not captchas — the cost of a wrong guess
should be a wait, not a lost signup. Password checks are metered twice (per
name *submitted* and per address, each stopping something the other can't),
deletion on its own key so an attack on an account's password can't block
its owner's erase button, and answering — the one unauthenticated write, and
the one that mints rows — per address, far above human pace. Limits are
charged before the slow work and never refunded; argon2 runs outside the
write transaction under a small concurrency cap. A guest row commits atomically
with the answer that earns it and foreign keys are enforced, so the only
garbage collection is a periodic sweep of expired session rows — user rows
are always history worth keeping. "Address" behind a proxy means a header
the proxy overwrites (`CLIENT_IP_HEADER`), never a forwarded header the
caller can seed, and the in-app counters are insurance — real per-IP volume
belongs in the proxy. The full reasoning for each choice sits beside the
limiters in `trainer/server.py` and on `RateLimiter` in `trainer/auth.py`.

Deletion runs on signup's reasoning in reverse: the signed-in session is the
proof of ownership (the optional email is never verified, so an email thread
could prove nothing), the password is re-entered anyway because a shared
browser holds the session, and responses and sessions are erased before the
`users` row they reference, all in one transaction. Guests have no password
and so can't be authenticated at all: clearing the cookie is the only
deletion available on a row nobody can point at. `trainer/account.py` is the
operator's way in for what the app can't reach — rows that predate accounts,
or a password nobody remembers — and doubles as the only recovery path, so
setting a password there also signs the account's existing sessions out.

Two ordering constraints are easy to get wrong. Identity resolution is a
dependency, but it yields a user *id*: FastAPI finishes dependencies before
the endpoint body — possibly on another thread, and so another connection —
so a row read there would be a snapshot from outside the endpoint's
transaction and concurrent answers would overwrite each other's ratings. And the session
cookie is applied by middleware (and by the catch-all error handler, which
sits outside it) rather than by the endpoint, so a request that mints a
guest and then fails still hands the identity out instead of orphaning it.
Signup resolves identity in its body rather than through the dependency, for
the same reason in reverse: a throttled request should mint nothing at all.

## Frontend (`web/`)

Two steps, and the split between them is the point. `scripts/build-web.mjs`
(esbuild) bundles and minifies `web/` into `web-dist/`, and does nothing else —
it renames nothing, rewrites no reference, and decides nothing about caching.
`trainer/assets.py` then reads whichever tree the server was pointed at, once at
startup, and does all of the semantics: it rewrites every reference to carry a
digest of what it points at, so an asset URL names its own contents and can be
cached forever, and it precompresses each one. So there is one implementation of
what a URL means, it reads the disk rather than a manifest that could disagree
with it, and the two trees differ only in size — which is what lets a dev
checkout serve the sources with no build and still be running the app the image
serves. CI runs the suite against both to hold them to it.

The pages carrying those digests are the only thing revalidated on a repeat
visit. Reaching an asset without its digest is still served, briefly cached: a
bookmark has no way to learn the file moved on. Compression is maximum-effort
because it happens once per boot rather than per request — about a tenth of a
second for the tree — and the encoding is part of the ETag, since it is part of
the body.

Bundling is arranged so no HTML changes between the trees: `board.css` exists
only to `@import` chessground's three stylesheets, which a browser follows on
its own and esbuild inlines. Chessground itself is a pinned npm dependency,
copied into `web/vendor/` by `scripts/vendor.mjs` on `npm ci` so the pages can
reach it at a URL; both that directory and `web-dist/` are output, and neither
is committed.

Vanilla JS, and no client-side chess logic — the server precomputes SAN and
per-ply FENs so the client only renders. Candidate moves are drawn as arrows;
answers by tap or keyboard; the reveal shows evals in centipawns and win
probability, auto-plays the chosen move's engine line (switchable to the other,
steppable, speed configurable), and can copy the position + both lines as plain
text for pasting into an assistant.

The palette lives entirely in `:root`, including the dark, saturated
arrow variants the light board needs alongside the light ones the dark
panel needs; `app.js` reads the arrow colours back out of CSS so a brush
can't drift away from the button it is supposed to match.

Every arrow carries a numbered disc, drawn by chessground's own shape
labels in the arrow's colour and repeated as a badge on the control it
pairs with: the choice buttons while choosing, the line cards after the
reveal. Each phase numbers whichever of those it is showing, so the discs
always agree with what pressing 1 and 2 does at that moment. The number is
the channel that survives two arrows crossing or landing on the same
square, where no pair of colours would have separated them — except when
they arrive along the same ray, a battery recapturing on one square, which
puts both discs at the same point and is not solved here. The drawer can
turn the numbers off, which takes the badges with them: without a disc to
pair with, the badge is only a keyboard hint again and is styled as one.

Colour is still what the eye reads first, so it carries the full weight.
The candidate pair is blue against orange, which sits on the
blue-yellow axis that red-green deficiencies leave intact, and it holds up
even with a warm night filter stacked on top. The reveal pair is green against
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
by `scripts/generate_assets.py` screenshotting Chromium, so the art has a
source that can be edited and re-run. The card mock declares no colours of its own: the generator injects
the app's `:root`, because a card advertising the app shouldn't be showing
a palette the app has since moved off. Every page carries its own copy of
the card metadata, since there is no template to share one; a test globs
`web/*.html` and holds each copy against the page it sits on, so a new page
can't ship without it.

## Storage

One SQLite database: `items` (positions, moves, evals, lines, difficulty),
`users` (rating, calibration state, optional credentials), `sessions`
(hashed cookie tokens), `responses` (every answer, timed, with rating
snapshots), `meta` (schema version, for the migrations that can't tell from
the data whether they already ran — everything else is guarded by a read).
The item bank is disposable and rebuildable from the pipeline —
literally so, since nothing the app does writes to it; responses are the
experimental record.

## Deployment (`deploy/`, `terraform/`)

One Fly machine with the database on a volume — SQLite has one writer, so a
second machine would be a second fork of the history rather than redundancy.
The image carries the server only; Stockfish and zstd belong to the
pipeline, which stays on a laptop. Litestream supervises uvicorn and streams
the file to S3 continuously, because a volume is one disk on one host and
`responses` can't be regenerated from anything. AWS holds the backup bucket,
Litestream's IAM user, and the DNS record, in Terraform; Fly's own provider
is archived, so that side is `fly.toml` and `flyctl`.

The consequence worth naming is that a refreshed item bank can't arrive as a
file: the bank and the record share a file, so `trainer/push_items.py`
carries items across as their own database and merges them in, matching on
position rather than on row id. Positions already present are skipped —
relabelling an item under the answers already given to it would make those
answers uninterpretable. The single exception is the lookahead ladder on a
position the live bank has never had measured, which the merge fills in along
with everything read off it: it needs Stockfish and so can only be measured on
the pipeline's side, and it changes how hard the item is *said* to be without
touching which move is correct. The merge writes the difficulty itself rather
than leaving it to `db.connect`, because the server it is merging under ran
its migrations at boot and will not run them again.

Bootstrap, runbooks (bank refresh, restore), and the operational cautions
live in `deploy/README.md`; everything else about the container is said in
the Dockerfile, `fly.toml`, and `deploy/entrypoint.sh` where it applies.
