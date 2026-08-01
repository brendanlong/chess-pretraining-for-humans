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

from . import assets, auth, db, rating, trials
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
# docstring. Per-process and lost on restart, costing at most one extra replay
# per token; the short anonymous token life keeps the set small.
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
        samesite="lax",  # blocks cross-site POSTs, which is our CSRF story
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


# Almost everything the app loads is its own: one module script, local
# stylesheets, a vendored chessground, and favicons as inline data: URIs (which
# is what `img-src data:` is for, along with the piece sprites in the
# chessground CSS). The reason to bother is that the reveal builds one string
# from mined game data — a CSP is what keeps a bad `Site` header in some future
# PGN from being script instead of a broken link.
#
# The page counter is the exception, and costs three entries: the script, plus
# the beacon under both `connect-src` (it uses `sendBeacon`) and `img-src` (it
# falls back to an image when that's refused). Scoped to the one path it posts
# to, because `connect-src` is otherwise a door out for the same hostile string
# the rest of this policy exists to contain.
ANALYTICS_SCRIPT = "https://gc.zgo.at"
ANALYTICS_BEACON = "https://chess-pretraining.goatcounter.com/count"
CSP = (
    f"default-src 'self'; script-src 'self' {ANALYTICS_SCRIPT}; style-src 'self'; "
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


def is_calibrating(user: dict) -> bool:
    return user["calib_step"] >= rating.CALIB_END_STEP


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


# A caller who has answered nothing has no row to read a rating or a history
# from, so both are the defaults a first trial would have used anyway: beginner
# rating, nothing seen yet. `user_id is None` is that caller throughout.
UNSEEN_COUNT_SQL = """SELECT COUNT(*) FROM items
    WHERE learnable = 1
      AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)"""


def unseen_count(user_id: int | None) -> int:
    # no row means no responses, so nothing is excluded
    return conn.execute(UNSEEN_COUNT_SQL, (user_id or 0,)).fetchone()[0]


# The pool one trial is drawn from (`trainer.supply` reports against it too).
SELECTION_POOL = 30

# The nearest unseen items on one side of the target. Two of these instead of
# one ORDER BY ABS(rating - ?) over the bank: the ABS ordering is one no index
# can provide, so it cost a scan of every item row plus a sort per trial —
# linear in the bank, and the dominant cost of serving. Each of these walks
# `idx_items_learnable_rating` away from the target until it has a pool's
# worth, so together they read about two pools of index entries (plus any the
# seen-item filter skips, checked on the index's rowid alone) however big the bank
# grows, and their union necessarily contains the pool nearest overall.
NEAREST_SQL = """SELECT * FROM items
    WHERE learnable = 1 AND rating {} ?
      AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)
    ORDER BY rating {} LIMIT {}"""
NEAREST_BELOW_SQL = NEAREST_SQL.format("<=", "DESC", SELECTION_POOL)
NEAREST_ABOVE_SQL = NEAREST_SQL.format(">", "ASC", SELECTION_POOL)


def pick_item(user_rating: float, user_id: int | None) -> tuple[dict | None, bool]:
    """An unseen item near the target difficulty; (item, is_repeat)."""
    target = rating.target_item_rating(user_rating)
    rows = sorted(
        conn.execute(NEAREST_BELOW_SQL, (target, user_id or 0)).fetchall()
        + conn.execute(NEAREST_ABOVE_SQL, (target, user_id or 0)).fetchall(),
        key=lambda row: abs(row["rating"] - target),
    )[:SELECTION_POOL]
    if rows:
        return dict(rng.choice(rows)), False
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


@app.get("/api/next")
def next_item(user_id: int | None = OptionalUserId):
    """Serve a trial. Writes nothing — a first-time visitor has no row yet, and
    getting one is what answering earns."""
    u = auth.get_user(conn, user_id) if user_id is not None else None
    item, is_repeat = pick_item(u["rating"] if u else rating.USER_START, user_id)
    if item is None:
        raise HTTPException(503, "no items in bank — run the mining/labeling pipeline")
    moves = [item["best_uci"], item["distractor_uci"]]
    rng.shuffle(moves)
    return {
        "item_id": item["id"],
        # The server's proof that it offered this item to this caller, which is
        # what /api/answer checks instead of consulting a row that may not exist.
        "trial_token": trials.issue(item["id"], user_id, is_repeat),
        "fen": item["fen"],
        "side_to_move": "white" if chess.Board(item["fen"]).turn else "black",
        "moves": [{"uci": m, "san": san(item["fen"], m)} for m in moves],
        "repeat": is_repeat,
        # No fresh-item count: it costs a pass over the bank, and the drawer
        # counter that reads it is seeded from /api/stats and counted down there.
        "trial_number": (u["attempts"] if u else 0) + 1,
        "user_rating": round(u["rating"] if u else rating.USER_START),
        "calibrating": is_calibrating(u) if u else True,
    }


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
            served_as_repeat = trials.redeem(a.trial_token, a.item_id, u["id"] if u else None)
        except trials.InvalidTrial as e:
            raise HTTPException(409, f"{e} — fetch a new trial") from e
        if u is None:
            # An anonymous token is the one kind a replay can profit from, because
            # the row that would notice the repeat doesn't exist yet. Answered as a
            # 409 rather than the limiter's 429: "this trial is spent" is the same
            # thing the client already knows how to recover from by fetching another.
            try:
                anonymous_trial_use.consume(a.trial_token or "")
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
        if is_repeat and not served_as_repeat:
            raise HTTPException(409, "that trial has already been answered — fetch a new one")
        # Repeats only happen when the bank is exhausted; they get feedback like
        # any trial but don't move the rating — a remembered answer isn't skill.
        new_step = u["calib_step"]
        if is_repeat:
            new_user_r = u["rating"]
        elif is_calibrating(u):
            new_user_r, new_step = rating.calibrate(u["rating"], u["calib_step"], correct)
        else:
            new_user_r = rating.update(u["rating"], item["rating"], correct)

        tx.execute(
            """INSERT INTO responses
               (user_id, item_id, choice_uci, correct, response_ms,
                user_rating_before, user_rating_after, item_rating_before)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                u["id"],
                item["id"],
                a.choice_uci,
                int(correct),
                a.response_ms,
                u["rating"],
                new_user_r,
                item["rating"],
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
        "calibrating": new_step >= rating.CALIB_END_STEP,
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
        "gap_wp": round(item["gap_wp"] * 100, 1),
        # What made this item as hard as it was rated, and the answer to "why
        # was that worth so much when the moves are miles apart": `gap_wp` is
        # what the answer is worth at full depth, `shallow_gap` is what there
        # was to see. Safe here and nowhere earlier — both are hints about where
        # to look, so they belong to the reveal and not to the trial.
        #
        # Nothing else off the ladder goes out. `gap_ladder` itself would be
        # answer-adjacent data sent for no reason: nothing on the page reads it,
        # and its signs are the answer key spelled out depth by depth.
        "shallow_gap": (
            None if item["shallow_gap"] is None else round(item["shallow_gap"] * 100, 1)
        ),
        "distractor_source": item["distractor_source"],
        "game_url": item["game_url"],
        "item_rating": round(item["rating"]),
    }


# Accuracy is over first exposures only: repeats, served once the bank is
# exhausted, can be answered from memory of the reveal rather than from skill.
#
# Newest-first with a limit, which is what keeps this endpoint a constant cost
# rather than one that grows with a user's history — the window is all anyone
# reads, so there is no reason to walk a career to compute it. The inner
# question ("is there an earlier answer to this item") must be answered from an
# index covering `item_id`; a test asserts that plan, because without it this
# reverts to quadratic. See `idx_responses_item` in `db.py`.
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
        "accuracy_last_50": round(sum(recent) / len(recent), 3) if recent else None,
        "items_remaining": unseen_count(user_id),
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
    password: str


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
    """Erase this account, its sessions, and every response it gave.

    Being signed in *is* the proof of ownership, which is why deletion belongs
    here rather than in an email thread: the optional email is never verified,
    so for most accounts there is no address a request could arrive from. The
    password is still asked for again, because a shared or unattended browser
    shouldn't be able to wipe someone's history from a drawer.

    A guest has no password to check and so can't be authenticated at all;
    clearing the cookie is what deletion means for a row nobody — us included
    — can point at, which is what the privacy policy says.

    Alone among the endpoints, this one resolves the session itself instead of
    taking `CurrentUserId`. Minting a guest is what that dependency does for a
    request without a cookie, and here it would mean writing two rows for a
    request we are about to refuse — a deletion request that arrives with no
    session has nothing to delete. Signup avoids the same trap the same way.
    """
    try:
        u = auth.session_user(conn, request.cookies.get(auth.COOKIE_NAME))
        if u is None or auth.is_guest(u):
            raise HTTPException(
                400,
                "There's no account here to delete. A guest record is reachable only "
                "through this browser's cookie, so clearing the cookie is what deleting "
                "it means — after that nobody, including us, can find it again.",
            )
        # Metered like login, because this endpoint checks a password too and
        # would otherwise be an unmetered guessing oracle. Charged before the
        # verify, for the reason spelled out at the limiters.
        #
        # Its own key, though, not login's. Reaching here already required a
        # signed-in session, so there is nothing to enumerate and no reason for
        # someone guessing at this account's password from outside to be able
        # to spend the budget its owner needs to erase it. Erasure is the one
        # promise the privacy policy makes that a user might need urgently.
        spend(login_ip_limiter, client_key(request))
        spend(delete_limiter, f"delete:{u['id']}")
        # Outside the transaction: argon2 is deliberately slow, and the write
        # lock is the one thing every writer contends for.
        if not auth.verify_password(auth.credential_for(u), body.password):
            raise HTTPException(400, "Wrong password.")
        with writing() as tx:
            # The row was read before the verify, so re-check that the hash we
            # matched is still the current one — a password rotated away
            # mid-request must not authorize destroying the account.
            current = tx.execute(
                "SELECT 1 FROM users WHERE id = ? AND password_hash = ?",
                (u["id"], u["password_hash"]),
            ).fetchone()
            if current is None:
                raise HTTPException(400, "Wrong password.")
            counts = auth.delete_user(tx, u["id"])
            # Deleting the sessions already revoked this cookie server-side;
            # clear it too so the browser lands on a fresh guest rather than
            # presenting a token for a row that no longer exists.
            queue_cookie(request, None)
        # Ids are reused by SQLite once the highest row goes, so don't leave a
        # spent counter behind for whoever gets this one next. Reachable only by
        # knowing the password, exactly as in login.
        delete_limiter.clear(f"delete:{u['id']}")
        login_limiter.clear(login_key(u["name"]))
    except auth.AuthError as e:
        raise auth_error(e) from e
    return {"deleted": True, "responses_deleted": counts["responses"]}


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
