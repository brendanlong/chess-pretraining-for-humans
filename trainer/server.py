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
# lock (small tool, contention is irrelevant). Reentrant because the identity
# dependency and the endpoint it feeds both take it.
conn = connect(DEFAULT_DB, check_same_thread=False)
db_lock = threading.RLock()

# Signup is the expensive-to-abuse one (it burns usernames); login is the one
# worth guessing at. Neither is a captcha — see auth.RateLimiter.
signup_limiter = auth.RateLimiter(limit=5, window_s=3600)
login_limiter = auth.RateLimiter(limit=10, window_s=900)


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with db_lock:
            return fn(*args, **kwargs)

    return wrapper


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


def current_user(request: Request, response: Response) -> dict:
    """Resolve the session cookie, minting a guest identity if there isn't one.

    Landing on the site is enough to start answering: no name to type, and
    the row is reachable only through an unguessable token rather than a
    guessable name in a URL.
    """
    with db_lock:
        user = auth.session_user(conn, request.cookies.get(auth.COOKIE_NAME))
        if user is None:
            user = auth.create_guest(conn, rating.USER_START, rating.CALIB_START_STEP)
            set_session_cookie(request, response, auth.start_session(conn, user["id"]))
        return user


CurrentUser = Depends(current_user)


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
def next_item(u: dict = CurrentUser):
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
def answer(a: Answer, u: dict = CurrentUser):
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
def stats(u: dict = CurrentUser):
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


def auth_error(e: auth.AuthError) -> HTTPException:
    return HTTPException(429 if isinstance(e, auth.RateLimited) else 400, str(e))


class Signup(BaseModel):
    username: str
    password: str
    email: str | None = None


class Login(BaseModel):
    username: str
    password: str


@app.get("/api/account")
def account(u: dict = CurrentUser):
    return account_payload(u)


@app.post("/api/account/signup")
@locked
def signup(body: Signup, request: Request, u: dict = CurrentUser):
    """Claim the guest row this session has been playing on — no reset."""
    try:
        signup_limiter.check(client_key(request))
        u = auth.claim(conn, u, body.username, body.password, body.email)
    except auth.AuthError as e:
        raise auth_error(e) from e
    return account_payload(u)


@app.post("/api/account/login")
@locked
def login(body: Login, request: Request, response: Response):
    try:
        login_limiter.check(client_key(request))
        u = auth.authenticate(conn, body.username, body.password)
    except auth.AuthError as e:
        raise auth_error(e) from e
    # Drop the session we arrived with (typically a fresh guest's) rather than
    # leaving a live token pointing at an abandoned row.
    auth.end_session(conn, request.cookies.get(auth.COOKIE_NAME))
    set_session_cookie(request, response, auth.start_session(conn, u["id"]))
    return account_payload(u)


@app.post("/api/account/logout")
@locked
def logout(request: Request, response: Response):
    auth.end_session(conn, request.cookies.get(auth.COOKIE_NAME))
    response.delete_cookie(auth.COOKIE_NAME, path="/")
    return {"ok": True}


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
