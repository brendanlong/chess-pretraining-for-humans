"""FastAPI app serving the discrimination trainer.

Run:
    uv run uvicorn trainer.server:app --host 0.0.0.0 --port <port>
"""

import functools
import random
import threading
from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import rating
from .db import DEFAULT_DB, connect
from .winprob import score_to_winprob

# Items are never repeated for a user: every trial is a first exposure, so
# the answer (recorded before feedback arrives) is an uncontaminated
# measurement AND the reveal can train on every trial. When the bank runs
# out, mine more games rather than recycling.
WEB_DIR = Path(__file__).resolve().parent.parent / "web"

app = FastAPI(title="chess-pretraining")
# FastAPI runs sync endpoints in a threadpool; share one connection behind a
# lock (single-user tool, contention is irrelevant).
conn = connect(DEFAULT_DB, check_same_thread=False)
db_lock = threading.Lock()


def locked(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with db_lock:
            return fn(*args, **kwargs)

    return wrapper


def get_user(name: str) -> dict:
    row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
    if row is None:
        conn.execute("INSERT INTO users (name) VALUES (?)", (name,))
        conn.commit()
        row = conn.execute("SELECT * FROM users WHERE name = ?", (name,)).fetchone()
    return dict(row)


def san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    return board.san(chess.Move.from_uci(uci))


def eval_display(cp: int | None, mate: int | None) -> str:
    if mate is not None:
        return f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    return f"{cp / 100:+.2f}"


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
def next_item(user: str = "default"):
    u = get_user(user)
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
    }


class Answer(BaseModel):
    item_id: int
    choice_uci: str
    response_ms: int | None = None
    user: str = "default"


@app.post("/api/answer")
@locked
def answer(a: Answer):
    u = get_user(a.user)
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
    if is_repeat:
        new_user_r, new_item_r = u["rating"], item["rating"]
    else:
        new_user_r, new_item_r = rating.update(u["rating"], item["rating"], correct)

    conn.execute(
        """INSERT INTO responses
           (user_id, item_id, choice_uci, correct, response_ms,
            user_rating_before, user_rating_after, item_rating_before, item_rating_after)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (u["id"], item["id"], a.choice_uci, int(correct), a.response_ms,
         u["rating"], new_user_r, item["rating"], new_item_r),
    )
    conn.execute(
        "UPDATE users SET rating = ?, attempts = attempts + 1 WHERE id = ?",
        (new_user_r, u["id"]),
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
        "correct": correct,
        "best": {
            "uci": item["best_uci"],
            "san": san(item["fen"], item["best_uci"]),
            "eval": eval_display(item["cp_best"], item["mate_best"]),
            "wp": round(item["wp_best"] * 100, 1),
        },
        "distractor": {
            "uci": item["distractor_uci"],
            "san": san(item["fen"], item["distractor_uci"]),
            "eval": eval_display(item["cp_distractor"], item["mate_distractor"]),
            "wp": round(item["wp_distractor"] * 100, 1),
        },
        "gap_wp": round(item["gap_wp"] * 100, 1),
        "distractor_source": item["distractor_source"],
        "game_url": item["game_url"],
        "item_rating": round(new_item_r),
    }


@app.get("/api/stats")
@locked
def stats(user: str = "default"):
    u = get_user(user)
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
    n_items, n_learnable = conn.execute(
        "SELECT COUNT(*), SUM(learnable) FROM items"
    ).fetchone()
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
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
