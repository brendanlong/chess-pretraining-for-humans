"""FastAPI app serving the discrimination trainer.

Run:
    uv run uvicorn trainer.server:app --host 0.0.0.0 --port <port>
"""

import contextlib
import hashlib
import logging
import os
import random
import sqlite3
import threading
from pathlib import Path

import chess
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from . import assets, auth, db, export, rating, trials
from .db import DEFAULT_DB, connect

log = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent


def _web_dir() -> Path:
    """The frontend tree to serve: the build's output if it was run, else the sources.

    The two differ only in size — `scripts/build-web.mjs` bundles and minifies
    and changes nothing else — so this can prefer whichever is there rather than
    having to be told. The image contains only `web-dist/`; a dev checkout has
    `web/` and needs no build to run. `TRAINER_WEB_DIR` is how the tests reach
    whichever one they aren't currently running against.
    """
    if override := os.environ.get("TRAINER_WEB_DIR"):
        return Path(override)
    built = _ROOT / "web-dist"
    return built if built.is_dir() else _ROOT / "web"


WEB_DIR = _web_dir()

# Which of the two moves is the correct one is decided by a coin flip, and that
# flip is the answer to the trial. The default `random` module is a Mersenne
# Twister whose state is recoverable from enough observed output, and a client
# observes the shuffle on every trial — so take the bit from the OS instead. A
# CSPRNG costs nothing here and removes the question entirely.
rng = random.SystemRandom()

app = FastAPI(title="Chess Pretraining")

# Which database the per-thread connections below open. A module-level name
# rather than a constant so the tests can point a whole server at a scratch file.
DB_PATH = DEFAULT_DB

# Migrations run once, here, on a connection nobody serves requests from.
connect(DB_PATH).close()

# One connection per thread, because a transaction belongs to a connection:
# sharing one would serialize every request, writing or not, and put two threads
# inside one transaction where SQLite cannot see them to object. Separate
# connections let WAL run readers alongside the single writer.
_threads = threading.local()


def thread_connection() -> sqlite3.Connection:
    existing = getattr(_threads, "conn", None)
    if existing is None:
        existing = db.open_connection(DB_PATH, explicit_transactions=True)
        _threads.conn = existing
    return existing


class OutsideTransaction(RuntimeError):
    """Raised for a use of the connection that `writing()` should have owned."""


class AmbientConnection:
    """The database outside any transaction: one statement, standing alone.

    `execute` and nothing else, because nothing else means anything here — a
    statement commits itself, and an open transaction belongs to the `writing()`
    block that opened it. Omitting them also omits `cursor()`, which would reach
    the connection underneath.

    Running a statement while a transaction is open raises instead, because that
    one is legal, looks like it works, and is how a block stops being atomic.
    """

    def execute(self, sql, parameters=(), /):
        connection = thread_connection()
        if connection.in_transaction:
            raise OutsideTransaction(
                "a transaction is open on this thread — use the handle "
                "`writing()` yielded instead of the module-level connection"
            )
        return connection.execute(sql, parameters)


conn: db.Queryable = AmbientConnection()

# Not a captcha (see auth.RateLimiter). Several limits, keyed on different
# things on purpose, because they defend different things.
#
# The password endpoints are metered twice over, and both halves are load-
# bearing. Per *name* is what protects one account's password: guessing from a
# hundred addresses is one line of script, so an address is not what attacks a
# password. Per *address* is what protects the box: argon2 is 64 MiB and ~50ms
# by design, `auth.HASH_CONCURRENCY` slots deep, so a few concurrent requests
# hold every slot and answer real users 503 — an unmetered password check is a
# denial-of-service primitive whether or not the name it names exists.
#
# Which is why the name key is the *submitted* name rather than the row it
# resolves to. Keying on a row id means a name nobody registered has no
# counter, and then the presence of a 429 is itself the answer to "does this
# account exist?" — an enumeration oracle that costs eleven requests and no
# credentials, and which the careful constant-time verify against a dummy hash
# was there to deny. Both cases hit the same counter with the same limit, so
# the boundary says nothing about which side of it you're on.
#
# Keying on a name the caller chooses does hand them the key space, which is
# what `RateLimiter._prune` guards; the per-address budget in front is what
# makes filling it expensive.
#
# The cost of per-name throttling is that a known account can be held locked
# by someone who keeps guessing at it. That is inherent, the short window is
# the mitigation, and deletion is deliberately keyed *separately* so the one
# irreversible thing a user might urgently need to do — the privacy policy's
# erase button — can't be blocked from outside by guessing at their password.
signup_limiter = auth.RateLimiter(limit=20, window_s=3600)
login_limiter = auth.RateLimiter(limit=10, window_s=900)
login_ip_limiter = auth.RateLimiter(limit=60, window_s=900)
delete_limiter = auth.RateLimiter(limit=10, window_s=900)
# Downloading your own record is neither a guess nor a write, so this rations
# work rather than attempts: one export walks a whole history and parses a FEN
# per answer, which is the only endpoint whose cost grows with how much
# somebody has played. Loose enough that fetching both formats, twice, changes
# nothing — and it is charged per address, because a session is free to mint.
export_limiter = auth.RateLimiter(
    limit=60,
    window_s=900,
    message="That's a lot of downloads at once. Try again in a few minutes.",
)
# Answering is the only unauthenticated write left — arriving costs nothing now —
# and it is the one that mints rows: a `responses` row every time, and a guest
# `users` row for a caller that arrives without one. Requiring a trial first
# can't gate that, since `next`→`answer` writes the same rows for one extra
# request. So this is where the volume gate belongs.
#
# Deliberately far above any human pace, because it sits on the core loop and
# several real users share one address routinely: 1200/15min is 80 answers a
# minute *aggregated over the address*, so it takes something like eight
# simultaneous fast players behind one NAT to notice it. That ceiling is the
# thing to raise if a shared address ever does. Like the others it is insurance,
# not a defence — real per-address volume is a reverse proxy's job.
answer_limiter = auth.RateLimiter(
    limit=1200,
    window_s=900,
    message="Answers are coming in faster than we can count. Try again in a moment.",
)
# Not a rate limit — a spend-once ledger, which is the same data structure. It
# is what keeps a spent anonymous trial spent, for the reasons in the `trials`
# docstring. Keyed on the trial's nonce rather than on the token carrying it, so
# that re-signing an expired token (`/api/trial/refresh`) shares the slot instead
# of getting a fresh one. Per-process and lost on restart, costing at most one
# extra replay per trial; the short anonymous token life keeps the set small.
anonymous_trial_use = auth.RateLimiter(
    limit=1,
    window_s=trials.ANON_TOKEN_TTL_S,
    message="That trial has already been answered — fetch a new one.",
)

# Expired sessions are swept every Nth guest minted rather than on a timer:
# there is no scheduler here, and new identities arriving is a decent clock
# for how fast the sessions table grows.
SWEEP_EVERY_GUESTS = 100
guests_minted = 0

# Which header, if any, carries an address the client can't choose for itself.
# Empty means "believe the socket", which is right when nothing is in front.
CLIENT_IP_HEADER = os.environ.get("CLIENT_IP_HEADER", "").lower()


def client_key(request: Request) -> str:
    """The address to charge a rate-limit slot to.

    Not `request.client.host`, when a proxy is in front. uvicorn's
    `--forwarded-allow-ips '*'` takes the *leftmost* `X-Forwarded-For` entry,
    and a proxy appends to whatever the client sent rather than replacing it —
    so that address is the caller's to choose, and a signup flood keyed on it
    would get a fresh counter per request. Name a header the proxy overwrites
    (`fly-client-ip` on Fly) and charge that instead.
    """
    if CLIENT_IP_HEADER:
        forwarded = request.headers.get(CLIENT_IP_HEADER)
        if forwarded:
            return forwarded
        # Configured but absent: we are being reached by something that didn't
        # come through the proxy, so the socket address is the caller's to
        # choose too. One shared key is a blunt answer, but it is the honest
        # one — better to over-throttle a path nothing legitimate uses than to
        # hand out a fresh counter per request.
        return "no-forwarded-header"
    return request.client.host if request.client else "unknown"


def auth_error(e: auth.AuthError) -> HTTPException:
    if isinstance(e, auth.RateLimited):
        return HTTPException(429, str(e))
    if isinstance(e, auth.AuthBusy):
        return HTTPException(503, str(e))
    return HTTPException(400, str(e))


def spend(limiter: auth.RateLimiter, key: str) -> None:
    """Take a rate-limit slot or refuse the request."""
    try:
        limiter.consume(key)
    except auth.AuthError as e:
        raise auth_error(e) from e


class Transaction:
    """What `writing()` hands out: a connection with no way to end it.

    SQLite does not count nesting, so a stray `commit()` is two different things
    depending on where it lands — nothing at all outside a transaction, and
    inside one the thing that ends it, dropping the write lock mid-request and
    putting the committed half beyond the reach of the rollback meant to undo it.
    A callee cannot tell which case it is in; its own writes make
    `in_transaction` true either way. So the method is absent rather than
    forbidden, and the block's outcome decides.
    """

    def __init__(self, connection):
        self._connection = connection

    def execute(self, sql, parameters=(), /):
        return self._connection.execute(sql, parameters)


@contextlib.contextmanager
def writing():
    """Everything inside commits together, or none of it does.

    `IMMEDIATE` rather than deferred because an answer reads a rating, computes
    the next in Python, and writes it back: taking the write lock up front makes
    the second of two overlapping answers wait, where deferred lets both read
    and then fails one at upgrade time.

    Held briefly and never across argon2 — every writer contends for it. Nesting
    raises, since SQLite has none: an inner block could only commit the outer one
    early or lie about being atomic.
    """
    connection = thread_connection()  # not `conn`, which refuses these on purpose
    if connection.in_transaction:
        raise OutsideTransaction("a transaction is already open on this thread")
    connection.execute("BEGIN IMMEDIATE")
    try:
        yield Transaction(connection)
        connection.commit()
    except BaseException:
        # Covers the commit too: it can fail on a full disk, and a transaction
        # left open is inherited by the next request on this thread, which then
        # cannot begin one for the life of the process.
        connection.rollback()
        raise


# --- identity -------------------------------------------------------------


def set_session_cookie(request: Request, response: Response, token: str) -> None:
    response.set_cookie(
        auth.COOKIE_NAME,
        token,
        max_age=auth.SESSION_DAYS * 86400,
        httponly=True,
        # Blocks cross-*site* POSTs, which is our CSRF story — and note the
        # word: a sibling subdomain of the apex is same-site and its requests
        # carry this cookie. Behind it stand two things that have to stay true,
        # because a guest's deletion has no password to fall back on: every
        # write takes a JSON body (a form post is a 422), and no CORS
        # middleware hands out permission to send one.
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def queue_cookie(request: Request, token: str | None) -> None:
    """Ask the middleware to set (or, for None, clear) the session cookie.

    Endpoints can't set it themselves: a guest minted while serving a request
    that then raises — an empty item bank answers 503 — would be committed to
    the database with its `Set-Cookie` discarded by the exception handler, and
    the client would mint another orphan on every retry.
    """
    request.state.session_cookie = "" if token is None else token


# Everything the app loads is its own: two module scripts, local stylesheets, a
# vendored chessground, and favicons as inline data: URIs (which is what
# `img-src data:` is for, along with the piece sprites in the chessground CSS).
# The reason to bother is that the reveal builds one string from mined game data
# — a CSP is what keeps a bad `Site` header in some future PGN from being script
# instead of a broken link.
#
# The page counter is a hosted service but not a hosted script: `web/count.js`
# builds the `/count` request itself, so `script-src` stays `'self'` and the
# counter costs two entries rather than three — the beacon under `connect-src`
# (`sendBeacon`) and under `img-src` (the fallback when that's refused). Scoped
# to the one path it posts to, because `connect-src` is otherwise a door out for
# the same hostile string the rest of this policy exists to contain. A test
# holds this constant against the one `count.js` posts to.
ANALYTICS_BEACON = "https://chess-pretraining.goatcounter.com/count"
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; "
    f"img-src 'self' data: {ANALYTICS_BEACON}; "
    f"connect-src 'self' {ANALYTICS_BEACON}; "
    "base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)


def finalize(request: Request, response: Response) -> Response:
    token = getattr(request.state, "session_cookie", None)
    if token:
        set_session_cookie(request, response, token)
    elif token == "":
        response.delete_cookie(auth.COOKIE_NAME, path="/")
    response.headers["Content-Security-Policy"] = CSP
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    if request.url.scheme == "https":
        # Only over https, and worth having because Fly's `force_https` is a
        # *redirect*: the first navigation is still a plaintext hop, which this
        # removes on every visit after it. No `includeSubDomains` — the app is
        # itself a subdomain and nothing lives under it, so that directive would
        # only commit names we don't have to a policy they don't need.
        response.headers["Strict-Transport-Security"] = "max-age=63072000"
    if request.url.path.startswith("/api/"):
        # Every API response is specific to one session's cookie, and one of
        # them hands out that cookie. A shared cache must never serve either.
        response.headers["Cache-Control"] = "no-store"
        response.headers["Vary"] = "Cookie"
    return response


@app.middleware("http")
async def session_cookie_middleware(request: Request, call_next):
    return finalize(request, await call_next(request))


@app.exception_handler(Exception)
def unhandled_error(request: Request, exc: Exception) -> Response:
    """Errors that escape the router are handled outside our middleware, so
    the cookie has to be re-applied here too — otherwise a crash while serving
    a brand-new visitor strands the guest it just created, once per retry."""
    return finalize(request, JSONResponse({"detail": "internal error"}, status_code=500))


# A write that waited out `busy_timeout` — another request, or a bank refresh
# merging into the live database. Nothing is wrong with the request, and 500
# means "don't bother trying again", which is the opposite of true here.
# Narrowed to SQLITE_BUSY: every other OperationalError is a bug, and inviting a
# retry on those invites hammering.
BUSY_MESSAGE = "The database is busy right now. Try again in a moment."


@app.exception_handler(sqlite3.OperationalError)
def database_busy(request: Request, exc: sqlite3.OperationalError) -> Response:
    if not (getattr(exc, "sqlite_errorname", "") or "").startswith("SQLITE_BUSY"):
        log.exception("database error serving %s", request.url.path, exc_info=exc)
        return unhandled_error(request, exc)
    log.warning("busy timeout serving %s", request.url.path)
    return finalize(request, JSONResponse({"detail": BUSY_MESSAGE}, status_code=503))


def optional_user_id(request: Request) -> int | None:
    """Resolve the session cookie to a user id, or None. Writes nothing.

    Identity is issued by *answering*, not by arriving — minting a row here
    would make arriving the cheapest write in the app, and metering that write
    puts a limit in front of the very first trial, which SPEC forbids. So a
    visitor who has answered nothing has no row, and the trial they are looking
    at is carried by a signed token instead (`trials`).

    Returns an *id*, not a row. FastAPI resolves sync dependencies in a
    separate threadpool call — possibly another thread, so another connection —
    that finishes before the endpoint body starts. A row read here is a snapshot
    from outside the endpoint's transaction, so endpoints that need one re-read
    it inside `writing()`.
    """
    token = request.cookies.get(auth.COOKIE_NAME)
    user = auth.session_user(conn, token)
    if user is None:
        return None
    auth.touch_session(conn, token)
    return user["id"]


OptionalUserId = Depends(optional_user_id)


def start_identity(tx: Transaction, request: Request) -> dict:
    """Create the row that answering earns, and hand its session out.

    The only place a `users` row is born from ordinary traffic. The caller owns
    the transaction: the row and its session commit together with the response
    that earns them, so a failure anywhere in between rolls the identity back
    (see `writing`) instead of leaving a row that answered nothing. The sweep
    rides along here, because arrival rate is the signal one is due; it is part
    of the same transaction, so it lands with the identity or not at all.
    """
    global guests_minted
    guests_minted += 1
    if guests_minted % SWEEP_EVERY_GUESTS == 1:
        auth.sweep(tx)
    user = auth.create_guest(tx, rating.USER_START, rating.CALIB_START_STEP)
    queue_cookie(request, auth.start_session(tx, user["id"]))
    return user


GUEST_ACCOUNT = {"username": None, "guest": True}


def account_payload(user: dict | None) -> dict:
    if user is None:
        return dict(GUEST_ACCOUNT)
    return {"username": auth.display_name(user), "guest": auth.is_guest(user)}


def san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    return board.san(chess.Move.from_uci(uci))


def eval_display(cp: int | None, mate: int | None) -> str:
    if mate is not None:
        return f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    assert cp is not None  # pov_parts always yields one of cp/mate
    return f"{cp / 100:+.2f}"


def line_steps(fen: str, pv: str | None, fallback_uci: str) -> list[dict]:
    """Engine line as replayable steps; each step's fen is the position after
    that ply, so the client can animate without any chess logic."""
    board = chess.Board(fen)
    steps = []
    for uci in (pv or fallback_uci).split():
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            break
        move_san = board.san(move)
        board.push(move)
        steps.append({"uci": uci, "san": move_san, "fen": board.fen()})
    return steps


# The pool one trial is drawn from (`trainer.supply` reports against it too).
SELECTION_POOL = 30

# The nearest unseen items on one side of the target: a walk along
# `idx_items_learnable_rating` away from it, stopping at a pool's worth. Two of
# these, merged, replace one ORDER BY ABS(rating - ?) over the bank — the ABS
# ordering is one no index can provide, so it cost a scan of every item row
# plus a sort per trial, linear in the bank; the union of the walks contains
# the pool nearest overall however big the bank grows. NOT EXISTS rather than
# NOT IN for the seen-item filter, because it probes `idx_responses_item` once
# per candidate — bounded by the walk — where NOT IN materializes the user's
# whole answer history every execution.
_NEAREST = """SELECT * FROM (
    SELECT * FROM items
     WHERE learnable = 1 AND rating {} :target
       AND NOT EXISTS (SELECT 1 FROM responses
                       WHERE user_id = :user_id AND item_id = items.id)
     ORDER BY rating {} LIMIT {})"""
_POOL = (
    _NEAREST.format("<=", "DESC", SELECTION_POOL)
    + " UNION ALL "
    + _NEAREST.format(">", "ASC", SELECTION_POOL)
)
# One row comes back: the random draw is the OFFSET, not a Python choice over a
# fetched pool. Each row a statement returns is its own sqlite3_step, releasing
# and retaking the GIL around a few microseconds of C — and under the
# threadpool's concurrency those round-trips convoy badly enough that fetching
# the pool cost more wall-clock than the bank scan it replaced. Everything up
# to the one row stays inside a single C call, which threads don't disturb.
PICK_SQL = f"SELECT * FROM ({_POOL}) ORDER BY ABS(rating - :target) LIMIT 1 OFFSET :k"
POOL_COUNT_SQL = f"SELECT COUNT(*) FROM ({_POOL})"


def pick_item(user_rating: float, user_id: int | None) -> tuple[dict | None, bool]:
    """An unseen item near the target difficulty; (item, is_repeat)."""
    target = rating.target_item_rating(user_rating)
    # A caller who has answered nothing has no row to read a rating or a history
    # from, so both are the defaults a first trial would have used anyway:
    # beginner rating, nothing seen yet. `user_id is None` is that caller
    # throughout, and no user 0 exists for the filters below to match.
    who = {"target": target, "user_id": user_id or 0}
    row = conn.execute(PICK_SQL, {**who, "k": rng.randrange(SELECTION_POOL)}).fetchone()
    if row is None:
        # Fewer unseen items left than a whole pool. Redraw over what remains,
        # which keeps the choice uniform — and the count is bounded by the same
        # two walks, so sizing it costs what the pick does.
        remaining = conn.execute(POOL_COUNT_SQL, who).fetchone()[0]
        if remaining:
            row = conn.execute(PICK_SQL, {**who, "k": rng.randrange(remaining)}).fetchone()
    if row is not None:
        return dict(row), False
    # Bank exhausted. Serve the least-recently-answered item so the app stays
    # usable, but flag it: repeat answers aren't clean measurements.
    row = conn.execute(
        """SELECT items.* FROM items
           JOIN responses ON responses.item_id = items.id
           WHERE items.learnable = 1 AND responses.user_id = ?
           GROUP BY items.id ORDER BY MAX(responses.id) LIMIT 1""",
        (user_id or 0,),
    ).fetchone()
    return (dict(row), True) if row else (None, False)


@app.get("/healthz")
def healthz():
    """Liveness for the platform's health check.

    Outside `/api/`, and free of the identity dependency and any transaction, so
    a slow query can't make a healthy machine look dead and have the proxy route
    around it mid-answer. It still shares the threadpool, which is what a
    genuinely saturated box shows up as. (A failed check does that and only that —
    Fly doesn't restart a machine over one.)
    """
    return {"ok": True}


def named_item(item_id: int) -> dict | None:
    """The item a URL names, if the bank has it to serve.

    Only two refusals, and both are the bank's rather than the caller's: an id
    from another bank, and an item the engine won't hold an answer to. Having
    answered it is not one of them. A URL is a durable thing that gets
    reopened — the tab reloads, somebody clicks the link twice, a friend sends
    back the link you sent them — and landing on a stranger's position instead
    of the one the link names is the wrong answer to all three. So it is served
    again, as a repeat: feedback, and a rating that doesn't move. That is what
    makes handing back a position whose answer the caller may already know
    safe, because the thing an answer may never be *aimed at* is the rating,
    and a repeat isn't aimed at anything.

    Nothing here is a leak the ordinary trial flow doesn't already allow: the
    payload is symmetric between the two moves, so naming an item buys the
    position and the pair, never which one is better. What it does buy is
    reaching an item selection would not have offered, and in bulk — the ids
    are sequential, so this is the bank's positions readable by counting. The
    answers stay behind `/api/answer`, which is where they were.
    """
    row = conn.execute("SELECT * FROM items WHERE id = ? AND learnable = 1", (item_id,)).fetchone()
    return dict(row) if row else None


def already_answered(item_id: int, user_id: int | None) -> bool:
    """Whether this caller has answered `item_id` — what makes a named item a
    repeat.

    Probes `idx_responses_item` at exactly (user, item) and stops at the first
    row, which is the same seek selection uses to skip a seen candidate. A
    count would read every earlier answer instead, on an endpoint nobody meters
    and over rows the caller is free to pile up.
    """
    return (
        conn.execute(
            "SELECT 1 FROM responses WHERE user_id = ? AND item_id = ? LIMIT 1",
            (user_id or 0, item_id),
        ).fetchone()
        is not None
    )


@app.get("/api/next")
def next_item(item: str | None = None, user_id: int | None = OptionalUserId):
    """Serve a trial. Writes nothing — a first-time visitor has no row yet, and
    getting one is what answering earns.

    `item` names a position rather than asking for one, which is what following
    somebody's link does. It is served instead of what selection would have
    picked, and the answer to it is marked — or, if the bank hasn't got it,
    selection picks after all and the caller can see it didn't get what it
    asked for, which is all the page needs to say so.
    """
    u = auth.get_user(conn, user_id) if user_id is not None else None
    # Text, parsed here, rather than declared an integer: a link travels through
    # chat clients and Markdown and comes back with a bracket on the end, and a
    # framework's validation error is not what somebody who followed one should
    # be looking at. So anything that isn't an id reads as no id, which is the
    # same fallback an id the bank can't serve gets.
    #
    # All three conditions earn their place. `isascii`, because `isdigit` alone
    # accepts other scripts' digits, which `int` then parses. And a length, on
    # both sides of `int`: past 18 digits SQLite can't bind the value, and past
    # 4300 Python won't parse the string at all — either one is a 500 on an
    # endpoint anyone can reach, which is a worse answer than the 422 this is
    # here to avoid.
    wanted = int(item) if item and item.isascii() and item.isdigit() and len(item) < 19 else None
    named = named_item(wanted) if wanted is not None else None
    is_repeat = False
    row = named
    if row is None:
        row, is_repeat = pick_item(u["rating"] if u else rating.USER_START, user_id)
    if row is None:
        raise HTTPException(503, "no items in bank — run the mining/labeling pipeline")
    # Asked only of a named item: selection reports its own repeats, and the
    # ordinary path already filtered on this history, so it must not pay again.
    if named is not None and already_answered(row["id"], user_id):
        is_repeat = True
    served = trials.Served(repeat=is_repeat, shared=named is not None)
    moves = [row["best_uci"], row["distractor_uci"]]
    rng.shuffle(moves)
    return {
        "item_id": row["id"],
        # The server's proof that it offered this item to this caller, which is
        # what /api/answer checks instead of consulting a row that may not exist.
        "trial_token": trials.issue(row["id"], user_id, served),
        "fen": row["fen"],
        "side_to_move": "white" if chess.Board(row["fen"]).turn else "black",
        "moves": [{"uci": m, "san": san(row["fen"], m)} for m in moves],
        "repeat": is_repeat,
        # Not which *kind* of repeat: the page asked for this position or it
        # didn't, so it can tell a reopened link from an exhausted bank by
        # whether it was handed what it named.
        "trial_number": (u["attempts"] if u else 0) + 1,
        "user_rating": round(u["rating"] if u else rating.USER_START),
        "calibrating": rating.is_calibrating(u["calib_step"]) if u else True,
    }


class TrialRefresh(BaseModel):
    item_id: int
    trial_token: str | None = None


@app.post("/api/trial/refresh")
def refresh_trial(r: TrialRefresh, user_id: int | None = OptionalUserId):
    """Re-sign a trial this caller is already holding, so that a token which
    outlived its expiry costs a round trip instead of the answer.

    The alternative — fetching a replacement trial — throws away a decision
    somebody had already made about a position they were still looking at, and
    the moment it does that most often is a first-time visitor's first answer,
    which is the one there is least reason to lose.

    It hands back a token and nothing else. That is what keeps it from being a
    second look at the item: a peek needs the position, the pair, or the answer,
    and none of them are here — everything in the reply was already in the token
    that had to be presented to get it. `/api/next` remains the only way to be
    told what a trial *is*.

    Unmetered for the same reason `/api/next` is: it mints no identity and
    records no answer, and it sits on the path to a first answer, which SPEC
    says nothing may gate. The answer it leads to is metered where every answer
    is.
    """
    try:
        token = trials.reissue(r.trial_token, r.item_id, user_id)
    except trials.InvalidTrial as e:
        # Including expiry, which is the case this exists for: `reissue` doesn't
        # consult the clock, so what reaches here is a token that was never ours,
        # names another item, or names another session.
        raise HTTPException(409, f"{e} — fetch a new trial") from e
    return {"item_id": r.item_id, "trial_token": token}


# Past an hour it is a parked tab, not a decision; negative is a broken clock.
# Client-supplied timing outside this range is recorded as "not measured"
# rather than believed or bounced — the answer itself is still real, and a 422
# over a timestamp would throw it away.
RESPONSE_MS_MAX = 3_600_000


class Answer(BaseModel):
    item_id: int
    choice_uci: str
    trial_token: str | None = None
    response_ms: int | None = None


@app.post("/api/answer")
def answer(a: Answer, request: Request):
    """Record an answer and reveal the engine's verdict.

    Resolves identity itself rather than through a dependency, because this is
    the endpoint that *creates* it: a row should exist only once someone has
    actually answered something, and only once we know the trial was ours.
    """
    spend(answer_limiter, client_key(request))
    # Read inside the transaction that writes it back: the rating and
    # calibration updates below are read-modify-write, so a row read before
    # the write lock was taken would let two overlapping answers clobber
    # each other.
    # Ends before the reveal is built: a failure there must not undo the answer,
    # nor the identity whose cookie the middleware has already promised.
    with writing() as tx:
        u = auth.session_user(tx, request.cookies.get(auth.COOKIE_NAME))
        # The token comes first, before anything that reads the item — because
        # *every* answer about an item is a fact about it. The response below is the
        # answer key outright, and even "that isn't one of the offered moves" tells
        # an id-counting caller which two moves an item pairs. Item ids are small
        # sequential integers, so nothing here may reflect one back without proof
        # that we served it.
        try:
            trial = trials.redeem(a.trial_token, a.item_id, u["id"] if u else None)
        except trials.TrialExpired as e:
            # The one refusal that isn't the end of this answer: the trial is
            # still this caller's, so `/api/trial/refresh` will re-sign it and
            # the same pick can be submitted again. Its own status, because the
            # client must not do that with any of the others — replaying a pick
            # against a session that changed under it files one person's answer
            # under another.
            raise HTTPException(410, f"{e} — refresh it and answer again") from e
        except trials.InvalidTrial as e:
            raise HTTPException(409, f"{e} — fetch a new trial") from e
        served = trial.served
        if u is None:
            # An anonymous token is the one kind a replay can profit from, because
            # the row that would notice the repeat doesn't exist yet. Answered as a
            # 409 rather than the limiter's 429: "this trial is spent" is the same
            # thing the client already knows how to recover from by fetching another.
            try:
                anonymous_trial_use.consume(trial.nonce)
            except auth.RateLimited as e:
                raise HTTPException(409, str(e)) from e

        item = tx.execute("SELECT * FROM items WHERE id = ?", (a.item_id,)).fetchone()
        if item is None:  # only reachable if the bank dropped it mid-trial
            raise HTTPException(404, "unknown item")
        item = dict(item)
        if a.choice_uci not in (item["best_uci"], item["distractor_uci"]):
            raise HTTPException(400, "choice is not one of the offered moves")

        if u is None:
            # Answering is what earns an identity. Nothing before this point wrote a
            # row, which is what keeps arriving free and the first trial ungated.
            u = start_identity(tx, request)

        correct = a.choice_uci == item["best_uci"]
        is_repeat = (
            tx.execute(
                "SELECT 1 FROM responses WHERE user_id = ? AND item_id = ? LIMIT 1",
                (u["id"], item["id"]),
            ).fetchone()
            is not None
        )
        # A repeat is legitimate only if we *offered* it as one, which the token
        # says. Deciding it from the bank here instead gets both boundaries wrong —
        # see the `trials` module docstring.
        #
        # So this is what keeps a *fresh* trial answerable once, and a token
        # offering a repeat is by construction exempt: it can be spent as often
        # as the answer limiter allows, for rows that rate nothing. Which is the
        # same ceiling fetching a new trial each time already reaches, so there
        # is nothing here for a ledger to buy.
        if is_repeat and not served.repeat:
            raise HTTPException(409, "that trial has already been answered — fetch a new one")
        # Repeats get feedback but move nothing: whether the bank ran out or a
        # link got reopened, they can be answered from memory of the reveal.
        #
        # A shared item on its first exposure moves the rating like any other —
        # it is a real first exposure and Elo already prices how hard it was.
        # The staircase is what can't take one: it steps by a fixed amount
        # *because* selection guarantees the item was aimed at the user, so on
        # an item nobody aimed it would pay a quarter of the scale for what, in
        # a two-alternative task, a beginner wins half the time by guessing.
        # Elo reads the item's difficulty, so it prices the same answer at a
        # whole K if the item was far above them and at nothing if it wasn't —
        # which is the question "are they better than we thought" actually
        # being asked.
        new_step = u["calib_step"]
        if is_repeat:
            new_user_r = u["rating"]
        elif rating.is_calibrating(u["calib_step"]) and not served.shared:
            new_user_r, new_step = rating.calibrate(u["rating"], u["calib_step"], correct)
        else:
            # At the K the answer count has earned: large while the rating
            # rests on a staircase's handful of answers, the settled K_USER
            # once a history stands behind it. `attempts` as read, before this
            # answer joins the count.
            new_user_r = rating.update(
                u["rating"], item["rating"], correct, rating.k_factor(u["attempts"])
            )

        tx.execute(
            """INSERT INTO responses
               (user_id, item_id, choice_uci, correct, response_ms,
                user_rating_before, user_rating_after, item_rating_before, shared,
                calibrating)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                u["id"],
                item["id"],
                a.choice_uci,
                int(correct),
                a.response_ms
                if a.response_ms is not None and 0 <= a.response_ms <= RESPONSE_MS_MAX
                else None,
                u["rating"],
                new_user_r,
                item["rating"],
                # Off the token, not off the request: what the client sends is
                # the answer, not the story of how it got the question.
                int(served.shared),
                # The staircase's state as scored — before this answer's own
                # update, so the last calibration answer records 1.
                int(rating.is_calibrating(u["calib_step"])),
            ),
        )
        tx.execute(
            "UPDATE users SET rating = ?, calib_step = ?, attempts = attempts + 1 WHERE id = ?",
            (new_user_r, new_step, u["id"]),
        )
    # Nothing else to write: the `items` row an answer is about is never touched,
    # so one user's answers can't move what another user is served.

    return {
        "repeat": is_repeat,
        "user_rating": round(new_user_r),
        "rating_delta": round(new_user_r - u["rating"], 1),
        "calibrating": rating.is_calibrating(new_step),
        "correct": correct,
        "best": {
            "uci": item["best_uci"],
            "san": san(item["fen"], item["best_uci"]),
            "eval": eval_display(item["cp_best"], item["mate_best"]),
            "wp": round(item["wp_best"] * 100, 1),
            "line": line_steps(item["fen"], item["pv_best"], item["best_uci"]),
        },
        "distractor": {
            "uci": item["distractor_uci"],
            "san": san(item["fen"], item["distractor_uci"]),
            "eval": eval_display(item["cp_distractor"], item["mate_distractor"]),
            "wp": round(item["wp_distractor"] * 100, 1),
            "line": line_steps(item["fen"], item["pv_distractor"], item["distractor_uci"]),
        },
        # How hard the item was, which is safe here and nowhere earlier: it is a
        # hint about where to look, so it belongs to the reveal and not to the
        # trial. It goes out alone. The measurement behind it (`shallow_gap` and
        # the `gap_ladder` it averages) is answer-adjacent data nothing reads —
        # the ladder's signs are the answer key spelled out depth by depth — and
        # `gap_wp` is the difference between the two `wp`s already above.
        "item_rating": round(item["rating"]),
        "distractor_source": item["distractor_source"],
        "game_url": item["game_url"],
    }


# Accuracy is over first exposures only: a repeat can be answered from memory
# of the reveal rather than from skill.
#
# Newest-first with a limit, which is what keeps this endpoint a constant cost
# rather than one that grows with a user's history — the window is all anyone
# reads, so there is no reason to walk a career to compute it. The inner
# question ("is there an earlier answer to this item") must be answered from an
# index covering `item_id`; a test asserts that plan, because without it this
# reverts to quadratic. See `idx_responses_item` in `db.py`.
#
# Mirrored by `ACC_WINDOW` in `web/app.js`, which keeps extending this window as
# trials are answered: seeded at one width and extended at another, the header
# would report over neither.
ACCURACY_WINDOW = 50
RECENT_FIRST_EXPOSURES_SQL = f"""
    SELECT r.correct
      FROM responses r
     WHERE r.user_id = ?
       AND NOT EXISTS (SELECT 1 FROM responses p
                       WHERE p.user_id = r.user_id
                         AND p.item_id = r.item_id AND p.id < r.id)
     ORDER BY r.id DESC
     LIMIT {ACCURACY_WINDOW}"""


@app.get("/api/stats")
def stats(user_id: int | None = OptionalUserId):
    u = auth.get_user(conn, user_id) if user_id is not None else None
    recent = [r["correct"] for r in conn.execute(RECENT_FIRST_EXPOSURES_SQL, (user_id or 0,))]
    return {
        "user_rating": round(u["rating"] if u else rating.USER_START),
        # Off the row already read above rather than a COUNT over `responses`:
        # the answer that appends a response is the same one that increments
        # this, so counting them again asks the database a question it has
        # already written down.
        "attempts": u["attempts"] if u else 0,
        # The window itself, oldest-first, rather than the fraction over it: the
        # client keeps answering after this call and has to extend the same
        # window to stay right. Handed a fraction it would have to start a fresh
        # one, and the first answer after a page load would read 0% or 100%.
        "accuracy_window": recent[::-1],
        "account": account_payload(u),
    }


# --- accounts -------------------------------------------------------------
#
# Auth is orthogonal to the trial flow: these payloads carry no item data, so
# none of them can leak which move is better.


def login_key(username: str) -> str:
    """The login counter's key: the name as typed, case-folded because that is
    how the lookup compares it. Names that resolve to the same row must not get
    two budgets, and a name that resolves to nothing must still get one."""
    return f"name:{username.strip().casefold()}"


class Signup(BaseModel):
    username: str
    password: str
    email: str | None = None


class Login(BaseModel):
    username: str
    password: str


class Deletion(BaseModel):
    # Optional because a guest has none to send; the endpoint decides which
    # rows that is allowed to erase.
    password: str | None = None


def reissue_session(tx: Transaction, request: Request, user_id: int) -> None:
    """Point this browser at `user_id` on a brand-new token.

    Rotating on every privilege change means a token planted before signup
    (over plain http, say) can't be riding along on the account afterwards.
    """
    # Both the token we arrived with and one minted for us moments ago by
    # start_identity (whose cookie we are about to overwrite).
    auth.end_session(tx, request.cookies.get(auth.COOKIE_NAME))
    auth.end_session(tx, getattr(request.state, "session_cookie", None))
    queue_cookie(request, auth.start_session(tx, user_id))


@app.get("/api/account")
def account(user_id: int | None = OptionalUserId):
    return account_payload(auth.get_user(conn, user_id) if user_id is not None else None)


@app.post("/api/account/signup")
def signup(body: Signup, request: Request):
    """Claim the guest row this session has been playing on, or start a fresh
    one for somebody who signed up before answering anything. No reset."""
    # Charged before anything else, so no request can buy work by failing, and
    # a burst can't walk past a counter that only the outcome increments.
    spend(signup_limiter, client_key(request))
    try:
        # Everything cheap first: a typo, a taken name, or an already-claimed
        # session must not cost an argon2 hash (~50ms and 64 MiB).
        username, email = auth.validate_signup(body.username, body.password, body.email)
        user_id = optional_user_id(request)
        with writing() as tx:
            if user_id is None:
                auth.check_name_free(tx, username)
            else:
                auth.check_claimable(tx, user_id, username)
        password_hash = auth.hash_password(body.password)  # slow; not in a transaction
        with writing() as tx:
            if user_id is None:
                u = auth.create_account(
                    tx,
                    username,
                    password_hash,
                    email,
                    rating.USER_START,
                    rating.CALIB_START_STEP,
                )
            else:
                u = auth.claim(tx, user_id, username, password_hash, email)
            reissue_session(tx, request, u["id"])
    except auth.AuthError as e:
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/login")
def login(body: Login, request: Request):
    name = body.username.strip()
    # Both counters are charged before the verify and before the lookup, and
    # both are charged whether or not the name exists — see the limiters for
    # why either omission is a hole rather than a nicety. Charged before the
    # work rather than incremented after it: a counter read before the ~50ms
    # hash and written after is one a concurrent burst walks straight past.
    spend(login_ip_limiter, client_key(request))
    spend(login_limiter, login_key(name))
    try:
        u = auth.find_by_username(conn, name)
        # Verify outside the transaction: argon2 is deliberately slow, and
        # holding the write lock through it would stall every other writer.
        # An unknown name still pays for a verify against a dummy hash, so the
        # timing says nothing either.
        if not auth.verify_password(auth.credential_for(u), body.password) or u is None:
            raise HTTPException(400, "Wrong username or password.")
        with writing() as tx:
            # The row was read before the verify; re-check that the credential
            # we matched is still the current one, so a password rotated away
            # mid-login (trainer.account set-password) can't open a session.
            current = tx.execute(
                "SELECT 1 FROM users WHERE id = ? AND password_hash = ?",
                (u["id"], u["password_hash"]),
            ).fetchone()
            if current is None:
                raise HTTPException(400, "Wrong username or password.")
            # Drop the session we arrived with (typically a fresh guest's)
            # rather than leaving a live token pointing at an abandoned row.
            reissue_session(tx, request, u["id"])
        # Only reachable by knowing the password, so an attacker can't use it
        # to reset the count — and forgetting it would only over-throttle.
        login_limiter.clear(login_key(name))
    except auth.AuthError as e:  # a saturated hasher; the guess never happened
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/logout")
def logout(request: Request):
    with writing() as tx:
        auth.end_session(tx, request.cookies.get(auth.COOKIE_NAME))
    queue_cookie(request, None)
    return {"ok": True}


@app.post("/api/account/delete")
def delete_account(body: Deletion, request: Request):
    """Erase this session's record: the row, its sessions, and every response.

    Holding the session is the proof of ownership, which is why deletion
    belongs here rather than in an email thread: the optional email is never
    verified, so for most accounts there is no address a request could arrive
    from. It is also the same authorization export runs on — being this session
    is what "reading your own record" means throughout the app, and erasing it
    is the other half of that record's story.

    An account is asked for its password on top, because a shared or unattended
    browser holds the session and shouldn't be able to wipe someone's history
    from a drawer. A guest has no password, and there is no second factor to
    invent for one: the cookie is the only handle on that row, for them and for
    us. So the cookie has to be enough, because the alternative is that the
    guest data we do hold has no erase button at all — and "clear the cookie"
    is not one. That makes the record unreachable, which is not the same as
    gone, and the responses would stay in the research data the privacy policy
    promises deletion takes them out of.

    Alone among the endpoints, this one resolves the session itself instead of
    taking `CurrentUserId`. Minting a guest is what that dependency does for a
    request without a cookie, and here it would mean writing two rows for a
    request we are about to refuse — a deletion request that arrives with no
    session has nothing to delete. Signup avoids the same trap the same way.
    """
    try:
        u = auth.session_user(conn, request.cookies.get(auth.COOKIE_NAME))
        if u is None:
            raise HTTPException(
                400,
                "There's nothing here to delete yet — answering a question is "
                "what starts a record.",
            )
        guest = auth.is_guest(u)
        # Per user, both ways round, and never per address: erasure is the one
        # promise the privacy policy makes that a user might need urgently, so
        # nobody else may be able to spend the budget it needs. An address is
        # shared — several real players behind one NAT is the case the answer
        # limiter is sized around — and a guest reaching here spent an answer to
        # exist at all, so the volume that mints these rows is already metered
        # upstream. What this bounds is retries against one row.
        spend(delete_limiter, f"delete:{u['id']}")
        if not guest:
            # The password check is metered like login's on top, because it is
            # one, and an unmetered argon2 is a denial-of-service primitive
            # whether or not the guess is right. Charged before the verify, for
            # the reason spelled out at the limiters.
            spend(login_ip_limiter, client_key(request))
            if not body.password:
                # The drawer only omits the password where the row has none, so
                # an empty one here means this browser's idea of itself is out
                # of date — it signed up somewhere else, and is showing a form
                # with no field to type into. Say that rather than "wrong
                # password", which is true and useless. It tells the caller
                # nothing `/api/account` doesn't, and they already hold this
                # account's session or they wouldn't be reading it.
                raise HTTPException(
                    400,
                    "This browser is signed in to an account now. Reload the page "
                    "and confirm with its password.",
                )
            # Outside the transaction: argon2 is deliberately slow, and the
            # write lock is the one thing every writer contends for.
            if not auth.verify_password(auth.credential_for(u), body.password):
                raise HTTPException(400, "Wrong password.")
        with writing() as tx:
            # The row was read before the branch above chose how to authorize
            # it, so re-check that what it authorized against is still current.
            # The hash catches a password rotated away mid-request, and — `IS`
            # rather than `=`, because a guest's is NULL — a guest row that a
            # signup in another tab just claimed, which must not be erased
            # without the password it now has. The name is what pins the *row*:
            # NULL matches every guest, and SQLite hands a deleted id straight
            # back out, so without it a concurrent delete of the highest row
            # could leave this one erasing whoever inherited the number.
            current = tx.execute(
                "SELECT 1 FROM users WHERE id = ? AND name = ? AND password_hash IS ?",
                (u["id"], u["name"], u["password_hash"]),
            ).fetchone()
            if current is None:
                raise HTTPException(
                    400,
                    "This browser signed in to an account while the request was in "
                    "flight, so nothing was deleted. Try again."
                    if guest
                    else "Wrong password.",
                )
            counts = auth.delete_user(tx, u["id"])
            # Deleting the sessions already revoked this cookie server-side;
            # clear it too so the browser lands on a fresh guest rather than
            # presenting a token for a row that no longer exists.
            queue_cookie(request, None)
        # Ids are reused by SQLite once the highest row goes, so don't leave a
        # spent counter behind for whoever gets this one next. Reaching here
        # took the row's own secret — its password, or for a guest the cookie
        # that is the only one it has — so clearing it helps nobody else.
        delete_limiter.clear(f"delete:{u['id']}")
        if not guest:
            login_limiter.clear(login_key(u["name"]))
    except auth.AuthError as e:
        raise auth_error(e) from e
    return {"deleted": True, "responses_deleted": counts["responses"]}


@app.get("/api/account/export")
def export_account(request: Request, format: str = "json"):
    """Hand this session's own record back, as a file.

    The session cookie is the whole authorization, exactly as it is for reading
    the same record through `/api/stats` — this endpoint adds no reach, only a
    format. No password, unlike deletion: the password there guards an
    irreversible act on a browser somebody may have walked away from, and
    asking for one here would put a wall in front of the one thing a guest —
    who has no password at all — most needs, which is a copy of what they'd
    lose by clearing the cookie.

    A caller with no session has no row and so nothing to export; that is the
    same "there is nothing here" that deletion answers, and it is said rather
    than answered with an empty file.
    """
    # Charged first, like every other limiter here: a request must not be able
    # to buy work by failing on its way to it.
    spend(export_limiter, client_key(request))
    fmt = format.strip().casefold()
    if fmt not in export.FORMATS:
        raise HTTPException(400, f"format must be one of: {', '.join(export.FORMATS)}")
    u = auth.session_user(conn, request.cookies.get(auth.COOKIE_NAME))
    if u is None:
        raise HTTPException(
            400,
            "There's nothing here to export yet — answering a question is what starts a record.",
        )
    body, media_type, name = export.build(conn, u, fmt)
    return Response(
        body,
        media_type=media_type,
        # The filename is ours and ASCII (`export.filename`), so the quoted
        # form is the whole story — no RFC 5987 encoding to get wrong.
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


# Read once at startup: the tree is a few hundred KB, and holding it means the
# digests in `assets` can't disagree with what gets served under them.
WEB = assets.build(WEB_DIR)


@app.get("/{path:path}")
@app.head("/{path:path}")
def static_file(path: str, request: Request) -> Response:
    """The frontend. Registered last, so the API routes above win.

    An entry point is revalidated on every visit and everything else is cached
    forever, which is only safe because everything else is reached through a URL
    that names its own contents.
    """
    asset = WEB.get("/" + path if path else "/")
    if asset is None:
        return JSONResponse({"detail": "not found"}, status_code=404)
    body, encoding = asset.negotiate(request.headers.get("accept-encoding", ""))
    # The encoding is part of the tag because it is part of the body. A cache
    # holding both copies has to be able to tell them apart, and `Vary` alone
    # only tells it to keep them separate, not which one it has.
    variant = f"-{encoding}" if encoding else ""
    tag = f'"{asset.digest or hashlib.sha256(asset.body).hexdigest()[:12]}{variant}"'
    headers = {
        "Cache-Control": asset.cache_control(request.query_params.get(assets.VERSION_PARAM)),
        "ETag": tag,
        # Set even on the copies with no variants to offer: a shared cache that
        # stored one without it would serve it to a client that asked for
        # something else, and which files have variants is not the client's to
        # know.
        "Vary": "Accept-Encoding",
    }
    if encoding:
        headers["Content-Encoding"] = encoding
    # The entry points are the ones that get here with a stale copy in hand, and
    # answering 304 saves sending a page to say it hasn't changed.
    if request.headers.get("if-none-match") == tag:
        return Response(status_code=304, headers=headers)
    return Response(body, media_type=asset.media_type, headers=headers)
