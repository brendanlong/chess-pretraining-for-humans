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

PROBE_EVERY = 8  # every Nth trial gives no feedback; those are the real metric
RECENT_EXCLUDE = 30  # don't re-serve an item seen in the last N trials
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


def is_probe(attempts: int) -> bool:
    return attempts % PROBE_EVERY == PROBE_EVERY - 1


def san(fen: str, uci: str) -> str:
    board = chess.Board(fen)
    return board.san(chess.Move.from_uci(uci))


def eval_display(cp: int | None, mate: int | None) -> str:
    if mate is not None:
        return f"#{mate}" if mate > 0 else f"#-{abs(mate)}"
    return f"{cp / 100:+.2f}"


def pick_item(user: dict) -> dict | None:
    target = rating.target_item_rating(user["rating"])
    recent = [
        r["item_id"]
        for r in conn.execute(
            "SELECT item_id FROM responses WHERE user_id = ? ORDER BY id DESC LIMIT ?",
            (user["id"], RECENT_EXCLUDE),
        )
    ]
    placeholders = ",".join("?" * len(recent)) or "NULL"
    rows = conn.execute(
        f"""SELECT * FROM items
            WHERE learnable = 1 AND id NOT IN ({placeholders})
            ORDER BY ABS(rating - ?) LIMIT 30""",
        (*recent, target),
    ).fetchall()
    if not rows:
        rows = conn.execute(
            "SELECT * FROM items WHERE learnable = 1 ORDER BY ABS(rating - ?) LIMIT 30",
            (target,),
        ).fetchall()
    return dict(random.choice(rows)) if rows else None


@app.get("/api/next")
@locked
def next_item(user: str = "default"):
    u = get_user(user)
    item = pick_item(u)
    if item is None:
        raise HTTPException(503, "no items in bank — run the mining/labeling pipeline")
    moves = [item["best_uci"], item["distractor_uci"]]
    random.shuffle(moves)
    return {
        "item_id": item["id"],
        "fen": item["fen"],
        "side_to_move": "white" if chess.Board(item["fen"]).turn else "black",
        "moves": [{"uci": m, "san": san(item["fen"], m)} for m in moves],
        # Deliberately no probe flag here: announcing "this one doesn't
        # count" before the answer would change behavior on exactly the
        # trials that are the metric.
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
    probe = is_probe(u["attempts"])
    # Probe trials are pure measurement: no rating movement at all, or the
    # delta (even seen one trial later) becomes a correctness oracle.
    if probe:
        new_user_r, new_item_r = u["rating"], item["rating"]
    else:
        new_user_r, new_item_r = rating.update(u["rating"], item["rating"], correct)

    conn.execute(
        """INSERT INTO responses
           (user_id, item_id, choice_uci, correct, probe, response_ms,
            user_rating_before, user_rating_after, item_rating_before, item_rating_after)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (u["id"], item["id"], a.choice_uci, int(correct), int(probe), a.response_ms,
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

    if probe:
        # No-feedback trial: recorded, but the reveal (and any rating info
        # that could stand in for it) is withheld.
        return {"probe": True}

    return {
        "probe": False,
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
    rows = [
        dict(r)
        for r in conn.execute(
            """SELECT correct, probe, user_rating_after, created_at
               FROM responses WHERE user_id = ? ORDER BY id""",
            (u["id"],),
        )
    ]
    # Feedback and probe trials are reported separately: mixing them makes
    # the headline accuracy incomparable with the frontend's live window,
    # and the probe-vs-feedback contrast is the interesting number.
    feedback = [r for r in rows if not r["probe"]]
    last50 = feedback[-50:]
    probes = [r for r in rows if r["probe"]]
    n_items, n_learnable = conn.execute(
        "SELECT COUNT(*), SUM(learnable) FROM items"
    ).fetchone()
    return {
        "user_rating": round(u["rating"]),
        "attempts": len(rows),
        "accuracy": round(sum(r["correct"] for r in feedback) / len(feedback), 3)
        if feedback
        else None,
        "accuracy_last_50": round(sum(r["correct"] for r in last50) / len(last50), 3)
        if last50
        else None,
        "probe_attempts": len(probes),
        "probe_accuracy": round(sum(r["correct"] for r in probes) / len(probes), 3)
        if probes
        else None,
        "rating_history": [round(r["user_rating_after"]) for r in rows],
        "items_total": n_items,
        "items_learnable": n_learnable or 0,
    }


app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
