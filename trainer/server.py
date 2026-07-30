"""FastAPI app serving the discrimination trainer.

Run:
    uv run uvicorn trainer.server:app --host 0.0.0.0 --port <port>
"""

import functools
import os
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

# Which of the two moves is the correct one is decided by a coin flip, and that
# flip is the answer to the trial. The default `random` module is a Mersenne
# Twister whose state is recoverable from enough observed output, and a client
# observes the shuffle on every trial — so take the bit from the OS instead. A
# CSPRNG costs nothing here and removes the question entirely.
rng = random.SystemRandom()

app = FastAPI(title="Chess Pretraining")
# FastAPI runs sync endpoints in a threadpool; share one connection behind a
# lock (small tool, contention is irrelevant).
conn = connect(DEFAULT_DB, check_same_thread=False)
db_lock = threading.Lock()

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
# Arriving is enough to write a `users` row and a `sessions` row, which makes
# the unauthenticated write path the cheapest one in the app.
#
# This is the only limit an ordinary stranger can meet, and SPEC says nothing
# gates the first trial — so it is deliberately far looser than the others, and
# the real bound on a flood is `auth.GUEST_TTL_HOURS`: rows with no answers are
# reclaimed within hours, so what an address can hold is its rate times that
# window (~1800 rows, under a megabyte) rather than everything it ever sent.
# Tightening this instead of the TTL would be choosing to turn away real
# first-time visitors behind one carrier NAT, which is the wrong trade — real
# per-address volume belongs in a reverse proxy anyway.
guest_limiter = auth.RateLimiter(
    limit=600,
    window_s=3600,
    message="Too many new visitors from your network right now. Try again in a few minutes.",
)

# Guests are swept periodically rather than on a timer; there is no scheduler
# here and arrival rate is exactly the signal that we need one. The counter is
# per-process, so this used to fire near every wake back when the deployment
# idled to zero; now it really is once per 100 guests, which on a quiet week is
# a while. Bounded either way — growth between sweeps is ~100 guests.
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


# Everything the app loads is its own: one module script, local stylesheets, a
# vendored chessground, and favicons as inline data: URIs (which is what
# `img-src data:` is for, along with the piece sprites in the chessground CSS).
# So the policy needs no allowlist, and the reason to bother is that the reveal
# builds one string from mined game data — a CSP is what keeps a bad `Site`
# header in some future PGN from being script instead of a broken link.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
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
    # Charged only on the minting path, so a returning session never spends a
    # slot; taken outside the lock because refusing is the cheap answer.
    spend(guest_limiter, client_key(request))
    with db_lock:
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
        return dict(rng.choice(rows)), False
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


@app.get("/healthz")
def healthz():
    """Liveness for the platform's health check.

    Deliberately outside `/api/`: it takes no identity dependency, so a probe
    every few seconds doesn't mint (and then sweep) a guest row, and it doesn't
    take the database lock, so a slow query can't make a healthy machine look
    dead and have the proxy route around it mid-answer. (A failed check does
    that and only that — Fly doesn't restart a machine over one.)
    """
    return {"ok": True}


@app.get("/api/next")
@locked
def next_item(user_id: int = CurrentUserId):
    u = auth.get_user(conn, user_id)
    item, is_repeat = pick_item(u)
    if item is None:
        raise HTTPException(503, "no items in bank — run the mining/labeling pipeline")
    moves = [item["best_uci"], item["distractor_uci"]]
    rng.shuffle(moves)
    # This is the item /api/answer will accept, and the only one. Recorded
    # before the payload goes out, so there is no window where a trial is on
    # screen but unanswerable.
    conn.execute("UPDATE users SET pending_item_id = ? WHERE id = ?", (item["id"], u["id"]))
    conn.commit()
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
    # An answer is only ever to the trial this user was last served. The
    # response below is the answer key — best move, both evals, both lines —
    # and item ids are small sequential integers, so without this an
    # unauthenticated caller could read the whole bank by counting, or read the
    # answer to the trial currently on their own screen before committing to
    # it, which is the one thing SPEC says nothing may do. It also stops
    # `items.attempts`/`correct`/`rating` — global, shared by every user, and
    # surviving account deletion — from being writable by anyone with curl.
    if a.item_id != u["pending_item_id"]:
        raise HTTPException(409, "no trial in progress for this item — fetch a new one")
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
        # Clearing the pending trial spends it: a resubmitted answer gets the
        # 409 above rather than a second rating movement for the same trial.
        """UPDATE users SET rating = ?, calib_step = ?, attempts = attempts + 1,
                            pending_item_id = NULL WHERE id = ?""",
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
    name = body.username.strip()
    # Both counters are charged before the verify and before the lookup, and
    # both are charged whether or not the name exists — see the limiters for
    # why either omission is a hole rather than a nicety. Charged before the
    # work rather than incremented after it: a counter read before the ~50ms
    # hash and written after is one a concurrent burst walks straight past.
    spend(login_ip_limiter, client_key(request))
    spend(login_limiter, login_key(name))
    try:
        with db_lock:
            u = auth.find_by_username(conn, name)
        # Verify outside the lock: argon2 is deliberately slow, and holding the
        # single database lock through it would stall every trial in flight.
        # An unknown name still pays for a verify against a dummy hash, so the
        # timing says nothing either.
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
        # Only reachable by knowing the password, so an attacker can't use it
        # to reset the count — and forgetting it would only over-throttle.
        login_limiter.clear(login_key(name))
    except auth.AuthError as e:  # a saturated hasher; the guess never happened
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/logout")
@locked
def logout(request: Request):
    auth.end_session(conn, request.cookies.get(auth.COOKIE_NAME))
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
        with db_lock:
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
        # Outside the lock: argon2 is deliberately slow and every trial in
        # flight shares that lock.
        if not auth.verify_password(auth.credential_for(u), body.password):
            raise HTTPException(400, "Wrong password.")
        with db_lock:
            # The row was read before the verify, so re-check that the hash we
            # matched is still the current one — a password rotated away
            # mid-request must not authorize destroying the account.
            current = conn.execute(
                "SELECT 1 FROM users WHERE id = ? AND password_hash = ?",
                (u["id"], u["password_hash"]),
            ).fetchone()
            if current is None:
                raise HTTPException(400, "Wrong password.")
            counts = auth.delete_user(conn, u["id"])
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


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
