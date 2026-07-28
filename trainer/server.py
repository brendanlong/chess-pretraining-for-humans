"""FastAPI app serving the discrimination trainer.

Run:
    uv run uvicorn trainer.server:app --host 0.0.0.0 --port <port>
"""

import functools
import random
import threading
from pathlib import Path

import chess
from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth, rating
from .db import DEFAULT_DB, connect

# Items are never repeated for a user: every trial is a first exposure, so
# the answer (recorded before feedback arrives) is an uncontaminated
# measurement AND the reveal can train on every trial. When the bank runs
# out, mine more games rather than recycling.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="chess-pretraining")
# FastAPI runs sync endpoints in a threadpool; share one connection behind a
# lock (small tool, contention is irrelevant).
conn = connect(DEFAULT_DB, check_same_thread=False)
db_lock = threading.Lock()

# One counter per endpoint, charged on every request, never refunded — not a
# captcha (see auth.RateLimiter). Deliberately loose: what it has to stop is a
# script, and what it must not do is punish someone fumbling a form. Anything
# finer-grained needs per-outcome accounting, which is where this file's bugs
# have come from and which buys very little on a self-hosted trainer.
signup_limiter = auth.RateLimiter(limit=20, window_s=3600)
login_limiter = auth.RateLimiter(limit=20, window_s=900)

# Guests are swept periodically rather than on a timer; there is no scheduler
# here and arrival rate is exactly the signal that we need one.
SWEEP_EVERY_GUESTS = 100
guests_minted = 0


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with db_lock:
            return fn(*args, **kwargs)

    return wrapper


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


def finalize(request: Request, response: Response) -> Response:
    token = getattr(request.state, "session_cookie", None)
    if token:
        set_session_cookie(request, response, token)
    elif token == "":
        response.delete_cookie(auth.COOKIE_NAME, path="/")
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


def current_user_id(request: Request) -> int:
    """Resolve the session cookie to a user id, minting a guest if there isn't
    one. Landing on the site is enough to start answering: no name to type,
    and the row is reachable only through an unguessable token rather than a
    guessable name in a URL.

    Returns an *id*, not a row. FastAPI resolves sync dependencies in a
    separate threadpool call that finishes before the endpoint body starts, so
    a row read here would be a snapshot from an already-released critical
    section — two overlapping answers would both write ratings derived from
    the same stale row. Endpoints re-read under their own lock.
    """
    global guests_minted
    with db_lock:
        user = auth.session_user(conn, request.cookies.get(auth.COOKIE_NAME))
        if user is not None:
            return user["id"]
        guests_minted += 1
        if guests_minted % SWEEP_EVERY_GUESTS == 1:
            auth.sweep(conn)
        user = auth.create_guest(conn, rating.USER_START, rating.CALIB_START_STEP)
        queue_cookie(request, auth.start_session(conn, user["id"]))
        return user["id"]


CurrentUserId = Depends(current_user_id)


def account_payload(user: dict) -> dict:
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


def unseen_count(user: dict) -> int:
    return conn.execute(
        """SELECT COUNT(*) FROM items
           WHERE learnable = 1
             AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)""",
        (user["id"],),
    ).fetchone()[0]


def pick_item(user: dict) -> tuple[dict | None, bool]:
    """An unseen item near the target difficulty; (item, is_repeat)."""
    target = rating.target_item_rating(user["rating"])
    rows = conn.execute(
        """SELECT * FROM items
           WHERE learnable = 1
             AND id NOT IN (SELECT item_id FROM responses WHERE user_id = ?)
           ORDER BY ABS(rating - ?) LIMIT 30""",
        (user["id"], target),
    ).fetchall()
    if rows:
        return dict(random.choice(rows)), False
    # Bank exhausted. Serve the least-recently-answered item so the app stays
    # usable, but flag it: repeat answers aren't clean measurements.
    row = conn.execute(
        """SELECT items.* FROM items
           JOIN responses ON responses.item_id = items.id
           WHERE items.learnable = 1 AND responses.user_id = ?
           GROUP BY items.id ORDER BY MAX(responses.id) LIMIT 1""",
        (user["id"],),
    ).fetchone()
    return (dict(row), True) if row else (None, False)


@app.get("/api/next")
@locked
def next_item(user_id: int = CurrentUserId):
    u = auth.get_user(conn, user_id)
    item, is_repeat = pick_item(u)
    if item is None:
        raise HTTPException(503, "no items in bank — run the mining/labeling pipeline")
    moves = [item["best_uci"], item["distractor_uci"]]
    random.shuffle(moves)
    return {
        "item_id": item["id"],
        "fen": item["fen"],
        "side_to_move": "white" if chess.Board(item["fen"]).turn else "black",
        "moves": [{"uci": m, "san": san(item["fen"], m)} for m in moves],
        "repeat": is_repeat,
        "items_remaining": unseen_count(u),
        "trial_number": u["attempts"] + 1,
        "user_rating": round(u["rating"]),
        "calibrating": is_calibrating(u),
    }


class Answer(BaseModel):
    item_id: int
    choice_uci: str
    response_ms: int | None = None


@app.post("/api/answer")
@locked
def answer(a: Answer, user_id: int = CurrentUserId):
    # Read the row here, inside the lock that also writes it back: rating and
    # calibration updates below are read-modify-write, so a snapshot taken
    # before the lock would let two overlapping answers clobber each other.
    u = auth.get_user(conn, user_id)
    item = conn.execute("SELECT * FROM items WHERE id = ?", (a.item_id,)).fetchone()
    if item is None:
        raise HTTPException(404, "unknown item")
    item = dict(item)
    if a.choice_uci not in (item["best_uci"], item["distractor_uci"]):
        raise HTTPException(400, "choice is not one of the offered moves")

    correct = a.choice_uci == item["best_uci"]
    is_repeat = (
        conn.execute(
            "SELECT 1 FROM responses WHERE user_id = ? AND item_id = ? LIMIT 1",
            (u["id"], item["id"]),
        ).fetchone()
        is not None
    )
    # Repeats only happen when the bank is exhausted; they get feedback like
    # any trial but don't move ratings — a remembered answer isn't skill.
    new_step = u["calib_step"]
    if is_repeat:
        new_user_r, new_item_r = u["rating"], item["rating"]
    elif is_calibrating(u):
        # Item ratings are frozen while the user's rating is unreliable.
        new_user_r, new_step = rating.calibrate(u["rating"], u["calib_step"], correct)
        new_item_r = item["rating"]
    else:
        new_user_r, new_item_r = rating.update(u["rating"], item["rating"], correct)

    conn.execute(
        """INSERT INTO responses
           (user_id, item_id, choice_uci, correct, response_ms,
            user_rating_before, user_rating_after, item_rating_before, item_rating_after)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            u["id"],
            item["id"],
            a.choice_uci,
            int(correct),
            a.response_ms,
            u["rating"],
            new_user_r,
            item["rating"],
            new_item_r,
        ),
    )
    conn.execute(
        "UPDATE users SET rating = ?, calib_step = ?, attempts = attempts + 1 WHERE id = ?",
        (new_user_r, new_step, u["id"]),
    )
    conn.execute(
        "UPDATE items SET rating = ?, attempts = attempts + 1, correct = correct + ? WHERE id = ?",
        (new_item_r, int(correct), item["id"]),
    )
    conn.commit()

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
        "distractor_source": item["distractor_source"],
        "game_url": item["game_url"],
        "item_rating": round(new_item_r),
    }


@app.get("/api/stats")
@locked
def stats(user_id: int = CurrentUserId):
    u = auth.get_user(conn, user_id)
    # Only first exposures count toward accuracy: repeats (served only once
    # the bank is exhausted) can be answered from memory of the reveal.
    rows = [
        dict(r)
        for r in conn.execute(
            """SELECT r.correct, r.user_rating_after
               FROM responses r
               WHERE r.user_id = ?
                 AND NOT EXISTS (SELECT 1 FROM responses p
                                 WHERE p.user_id = r.user_id
                                   AND p.item_id = r.item_id AND p.id < r.id)
               ORDER BY r.id""",
            (u["id"],),
        )
    ]
    total_attempts = conn.execute(
        "SELECT COUNT(*) FROM responses WHERE user_id = ?", (u["id"],)
    ).fetchone()[0]
    last50 = rows[-50:]
    n_items, n_learnable = conn.execute("SELECT COUNT(*), SUM(learnable) FROM items").fetchone()
    return {
        "user_rating": round(u["rating"]),
        "attempts": total_attempts,
        "first_exposures": len(rows),
        "accuracy": round(sum(r["correct"] for r in rows) / len(rows), 3) if rows else None,
        "accuracy_last_50": round(sum(r["correct"] for r in last50) / len(last50), 3)
        if last50
        else None,
        "rating_history": [round(r["user_rating_after"]) for r in rows],
        "items_total": n_items,
        "items_learnable": n_learnable or 0,
        "items_remaining": unseen_count(u),
        "account": account_payload(u),
    }


# --- accounts -------------------------------------------------------------
#
# Auth is orthogonal to the trial flow: these payloads carry no item data, so
# none of them can leak which move is better.


def client_key(request: Request) -> str:
    # Behind a reverse proxy this is only the real client if uvicorn runs with
    # --proxy-headers and a trusted --forwarded-allow-ips.
    return request.client.host if request.client else "unknown"


def spend(limiter: auth.RateLimiter, ip: str) -> None:
    """Take a rate-limit slot or refuse the request."""
    try:
        limiter.consume(ip)
    except auth.AuthError as e:
        raise auth_error(e) from e


def auth_error(e: auth.AuthError) -> HTTPException:
    if isinstance(e, auth.RateLimited):
        return HTTPException(429, str(e))
    if isinstance(e, auth.AuthBusy):
        return HTTPException(503, str(e))
    return HTTPException(400, str(e))


class Signup(BaseModel):
    username: str
    password: str
    email: str | None = None


class Login(BaseModel):
    username: str
    password: str


def reissue_session(request: Request, user_id: int) -> None:
    """Point this browser at `user_id` on a brand-new token.

    Rotating on every privilege change means a token planted before signup
    (over plain http, say) can't be riding along on the account afterwards.
    """
    # Both the token we arrived with and one minted for us moments ago by
    # current_user_id (whose cookie we are about to overwrite).
    auth.end_session(conn, request.cookies.get(auth.COOKIE_NAME))
    auth.end_session(conn, getattr(request.state, "session_cookie", None))
    queue_cookie(request, auth.start_session(conn, user_id))


@app.get("/api/account")
@locked
def account(user_id: int = CurrentUserId):
    return account_payload(auth.get_user(conn, user_id))


@app.post("/api/account/signup")
def signup(body: Signup, request: Request):
    """Claim the guest row this session has been playing on — no reset."""
    # Charged before anything else, so no request can buy work by failing, and
    # a burst can't walk past a counter that only the outcome increments.
    spend(signup_limiter, client_key(request))
    try:
        # Everything cheap first: a typo, a taken name, or an already-claimed
        # session must not cost an argon2 hash (~50ms and 64 MiB).
        username, email = auth.validate_signup(body.username, body.password, body.email)
        # Identity is resolved here rather than by a dependency: dependencies
        # run before the body, so a throttled flood would still mint a guest
        # row per rejected attempt.
        user_id = current_user_id(request)
        with db_lock:
            auth.check_claimable(conn, user_id, username)
        password_hash = auth.hash_password(body.password)  # slow; not under the lock
        with db_lock:
            u = auth.claim(conn, user_id, username, password_hash, email)
            reissue_session(request, u["id"])
    except auth.AuthError as e:
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/login")
def login(body: Login, request: Request):
    # Charged before the verify, not incremented after it: a limit read before
    # the ~50ms hash and written after is one a concurrent burst walks straight
    # past. Every attempt counts, right or wrong — 20 per 15 minutes is far
    # more than a person signing in needs, including several sharing an address
    # (NAT, or a proxy started without --proxy-headers).
    spend(login_limiter, client_key(request))
    try:
        with db_lock:
            u = auth.find_by_username(conn, body.username.strip())
        # Verify outside the lock: argon2 is deliberately slow, and holding the
        # single database lock through it would stall every trial in flight.
        if not auth.verify_password(auth.credential_for(u), body.password) or u is None:
            raise HTTPException(400, "Wrong username or password.")
        with db_lock:
            # The row was read before the verify; re-check that the credential
            # we matched is still the current one, so a password rotated away
            # mid-login (trainer.account set-password) can't open a session.
            current = conn.execute(
                "SELECT 1 FROM users WHERE id = ? AND password_hash = ?",
                (u["id"], u["password_hash"]),
            ).fetchone()
            if current is None:
                raise HTTPException(400, "Wrong username or password.")
            # Drop the session we arrived with (typically a fresh guest's)
            # rather than leaving a live token pointing at an abandoned row.
            reissue_session(request, u["id"])
    except auth.AuthError as e:  # a saturated hasher; the guess never happened
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/logout")
@locked
def logout(request: Request):
    auth.end_session(conn, request.cookies.get(auth.COOKIE_NAME))
    queue_cookie(request, None)
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
